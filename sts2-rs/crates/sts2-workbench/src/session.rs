//! The branching, time-travel live session.
//!
//! Port of the M1 subset of `sts2-pycli/python/workbench/branching.py`
//! (`BranchingLiveSession`): one [`Game`] plus a **branch tree** of decision
//! points. The user can rewind to any past node and continue from there on
//! a new branch, keeping the original line intact.
//!
//! Restore is **replay-based** (spec v1): the engine is deterministic given
//! `(character, seed, ascension, act1)` plus an action sequence, so to
//! position the game at any node we rebuild the run and re-apply the
//! recorded actions along the path from the root. No mid-combat snapshots
//! are used — that keeps the "replay diverged" surface (a save that no longer
//! fits the engine) identical to Python's.
//!
//! Run-mode / autobattle / advisor orchestration / PPO value capture are
//! later milestones; every node's `value` is `null` here.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};
use sts2_core::{CombatState, GameVersion};

use crate::combatlog::{format_combat_log, LogNames};
use crate::game::{Game, GameError};
use crate::localization::{resolve_name, resolve_state_names, Loc};
use crate::playlog::format_play_log;
use crate::ppo::{self, Net};
use crate::render::{describe_action, NATIVE_ACTION};
use crate::runmode::RunAgent;
use crate::statediff::diff_states;

/// Actions the UI is allowed to send (`ALLOWED_ACTIONS`).
pub const ALLOWED_ACTIONS: &[&str] = &[
    "play_card",
    "end_turn",
    "choose_option",
    "select_map_node",
    "select_card_reward",
    "skip_card_reward",
    "buy_card",
    "buy_relic",
    "buy_potion",
    "remove_card",
    "select_bundle",
    "select_cards",
    "skip_select",
    "use_potion",
    "discard_potion",
    "leave_room",
    "proceed",
];

/// The default advisor of a hand-played run: the combat MCTS + the run
/// PPO / heuristic ([`Session::advisor`]).
pub const ADVISOR_SEARCH: &str = "search";
/// The e2e Ironclad student as the advisor (`session_e2e.rs`).
pub const ADVISOR_E2E: &str = "e2e";

/// Combat-log kinds kept in a node's tape for the VFX stream only
/// (`_VFX_ONLY_LOG_KINDS`).
const VFX_ONLY_LOG_KINDS: &[&str] = &["power_flash"];

/// What the Python session raised, with the HTTP status the server mapped
/// it to: `ValueError` → 400, `KeyError` (unknown node) → 404,
/// `RuntimeError` (no live run / replay diverged) → 409.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionError {
    pub status: u16,
    pub message: String,
}

impl SessionError {
    pub(crate) fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SessionError {}

/// One recorded step of the branch tree (`_Node`).
#[derive(Debug, Clone)]
pub struct Node {
    pub id: u64,
    pub parent_id: Option<u64>,
    /// Depth from the root (root = 0).
    pub idx: usize,
    pub decision: String,
    pub state: Value,
    /// The verb that produced this node from its parent (`None` on the root).
    pub action: Option<String>,
    pub args: Value,
    pub label: String,
    pub details: Vec<String>,
    /// This action's slice of the combat log, rendered for display.
    pub combat_log: Vec<Value>,
    pub error: bool,
    pub child_ids: Vec<u64>,
    /// PPO critic value (M3); always `None` in M1.
    pub value: Option<f64>,
    /// The engine's combat state at this node (`None` off combat). In
    /// memory only — never persisted; a restored session rebuilds it by
    /// replay. The advisors score a PAST position from this (the regret
    /// coach re-searches the parent of the move just made) instead of the
    /// lossy wire-dict round-trip the Python session had to make because it
    /// no longer held the parent's native state.
    pub combat: Option<CombatState>,
}

impl Node {
    /// Lightweight metadata for the timeline (no full state).
    #[must_use]
    pub fn meta(&self) -> Value {
        json!({
            "id": self.id,
            "parent_id": self.parent_id,
            "idx": self.idx,
            "decision": self.decision,
            "label": self.label,
            "details": self.details,
            "combat_log": self.combat_log,
            "error": self.error,
            "child_ids": self.child_ids,
            "value": self.value,
        })
    }
}

/// `apply()`'s result: the new (or re-synced) state plus the log label.
#[derive(Debug, Clone)]
pub struct ApplyResult {
    pub state: Value,
    pub error: Option<String>,
    pub label: String,
    pub details: Vec<String>,
}

/// `_progress`: leaf context for the save-list display.
#[derive(Debug, Clone, PartialEq)]
pub struct Progress {
    pub act: Value,
    pub floor: Value,
    pub hp: Value,
    pub max_hp: Value,
}

impl Progress {
    #[must_use]
    pub fn to_value(&self) -> Value {
        json!({"act": self.act, "floor": self.floor, "hp": self.hp, "max_hp": self.max_hp})
    }
}

/// `BranchingLiveSession` (M1 subset).
#[derive(Debug)]
pub struct Session {
    pub(crate) game: Option<Game>,
    pub character: String,
    pub seed: String,
    pub ascension: u8,
    /// Which Act-1 model this run walks (GH #544); `seek` rebuilds the run
    /// from scratch, so it is remembered, not re-derived.
    pub act1: String,
    /// The game build this run is played in (GH #801). Every constructor and
    /// rebuild (`spawn_game`) hands it to the [`Game`]; saves, the persist
    /// payload and the run log carry it when it is not the default.
    pub game_version: GameVersion,
    pub(crate) nodes: BTreeMap<u64, Node>,
    next_id: u64,
    root_id: Option<u64>,
    pub(crate) active_id: Option<u64>,
    pub(crate) loc: &'static Loc,
    /// True iff this run was created by `start_run_mode` — a Neow-start,
    /// integer-seed run that `seek` / a restore must rebuild through
    /// `start_faithful` (`_run_mode`).
    pub(crate) run_mode_run: bool,
    /// True iff this run-mode run was built as the e2e student's training run
    /// (`Game::start_e2e`: Ironclad v0.111.0, act 1 rolled, potion swap on).
    /// Part of the run's identity: `seek`, a restore and a save rebuild
    /// through that constructor.
    pub(crate) e2e_setup: bool,
    /// The e2e brain's net and per-node student state (`session_e2e.rs`).
    #[cfg(feature = "e2e")]
    pub(crate) e2e: crate::session_e2e::E2eSlot,
    /// A millisecond clock for the e2e brain's per-decision timing
    /// (`performance.now()` in the browser); `None` = the crate's fallback.
    pub(crate) perf_clock: Option<fn() -> f64>,
    /// The native combat-MCTS agent (`_faithful_agent`); built by
    /// `start_run_mode`, or lazily by the first `run_mode_step` on a
    /// hand-played run.
    pub(crate) agent: Option<RunAgent>,
    /// Cumulative agent-applied actions (`_faithful_steps`).
    pub(crate) faithful_steps: u64,
    // Two-axis run mode: who resolves off-combat decisions × combat budget.
    pub off_combat_brain: String,
    /// Who advises a hand-played move (`hint`, `auto-action`, the policy
    /// panel, the choice ranker, the choice coach, the map route): `search`
    /// (the combat MCTS + the run PPO / heuristic, the default) or `e2e`
    /// (the e2e Ironclad student, `session_e2e.rs`).
    pub advisor: String,
    pub ppo_impl: String,
    pub combat_sims: u32,
    pub combat_boss_sims: u32,
    pub combat_threads: u32,
    pub reward_sims: Option<i64>,
    // Autobattle: the state at the previous stop, the champion's withheld
    // pick at the current one, decisions auto-played since the last stop.
    pub(crate) autobattle_prev_state: Option<Value>,
    pub(crate) autobattle_pending: Option<Value>,
    pub(crate) autobattle_autoplayed: u64,
    /// The policy + run at the last human run-level choice, taken before it
    /// applied (`session_choice`): what the choice coach grades against.
    pub(crate) last_choice: Option<crate::session_choice::ChoiceSnapshot>,
    /// Pre-recorded node values consumed by `add_node` instead of the
    /// critic during a restore (`_restore_values`), and the read cursor.
    pub(crate) restore_values: Option<Vec<Option<f64>>>,
    pub(crate) restore_idx: usize,
    /// The loaded PPO champion for this run's character, and why there is
    /// none (`ppo::missing_net_error` when unset).
    pub(crate) ppo: Option<Net>,
    pub(crate) ppo_error: Option<String>,
    /// The pasted run position this session was built from, when it was
    /// created by `start_imported`. An imported run is NOT reproducible from
    /// `(character, seed, ascension, act1)` alone — the deck, relics, potions
    /// and RNG cursors are grafted — so `seek`, a reload and a named save must
    /// all rebuild it through the graft. `None` for every other run.
    pub(crate) imported: Option<crate::state_import::RunSnapshot>,
    /// The hint search's rows, PV and sims for each combat node it scored, by
    /// node id: the per-turn coach (`advisors::last_turn_grade`) grades a
    /// move against the very numbers the player was shown instead of
    /// searching the position again — when that search was at least as
    /// strong as the coach's own. In memory only.
    pub(crate) hint_scores: BTreeMap<u64, (Vec<Value>, Vec<Value>, u32)>,
}

impl Session {
    #[must_use]
    pub fn new(loc: &'static Loc) -> Self {
        Self {
            game: None,
            character: String::new(),
            seed: String::new(),
            ascension: 0,
            act1: "overgrowth".to_string(),
            game_version: GameVersion::default(),
            nodes: BTreeMap::new(),
            next_id: 0,
            root_id: None,
            active_id: None,
            loc,
            run_mode_run: false,
            e2e_setup: false,
            #[cfg(feature = "e2e")]
            e2e: crate::session_e2e::E2eSlot::default(),
            perf_clock: None,
            agent: None,
            faithful_steps: 0,
            off_combat_brain: "ppo".to_string(),
            advisor: ADVISOR_SEARCH.to_string(),
            ppo_impl: "rust".to_string(),
            // Benchmark (1000 / 4000) — the SPA's default preset, the offline
            // win-rate sweep's budget and `RunAgent`'s own default. It used to
            // be Quick (200 / 800), so any path that reached the agent without
            // the client having posted a budget played the run out at a fifth
            // of the budget the UI displayed.
            combat_sims: 1000,
            combat_boss_sims: 4000,
            combat_threads: 1,
            reward_sims: None,
            autobattle_prev_state: None,
            autobattle_pending: None,
            autobattle_autoplayed: 0,
            last_choice: None,
            restore_values: None,
            restore_idx: 0,
            ppo: None,
            ppo_error: None,
            imported: None,
            hint_scores: BTreeMap::new(),
        }
    }

    /// A fresh session in game build `version`, rendering with that build's
    /// embedded localization.
    #[must_use]
    pub fn new_for(version: GameVersion) -> Self {
        let mut sess = Self::new(Loc::embedded_for(version));
        sess.game_version = version;
        sess
    }

    /// Attach (or detach) the PPO champion this run scores with, and the
    /// reason there is none. Values are computed from the moment a net is
    /// attached; nodes created before keep `null`.
    pub fn set_ppo(&mut self, net: Option<Net>, error: Option<String>) {
        self.ppo = net;
        self.ppo_error = error;
    }

    /// Whether a champion net is attached.
    #[must_use]
    pub fn has_ppo(&self) -> bool {
        self.ppo.is_some()
    }

    /// The reason no net is attached (`no PPO net loaded for …`).
    #[must_use]
    pub(crate) fn net_error(&self) -> String {
        self.ppo_error
            .clone()
            .unwrap_or_else(|| ppo::missing_net_error(&self.character, None))
    }

    /// The backend name a save records.
    #[must_use]
    pub fn backend(&self) -> &'static str {
        "rust"
    }

    /// Whether this run was started run-mode (`start_run_mode`: integer
    /// seed, the faithful constructor). A save records it: it selects
    /// which constructor a restore rebuilds through.
    #[must_use]
    pub fn run_mode(&self) -> bool {
        self.run_mode_run
    }

    /// Whether the e2e student advises this run's hand-played moves.
    #[must_use]
    pub fn advisor_is_e2e(&self) -> bool {
        self.advisor == ADVISOR_E2E
    }

    /// Pick who advises: [`ADVISOR_SEARCH`] or [`ADVISOR_E2E`]. 400 for an
    /// unknown name, a build without the e2e feature, or a run the student
    /// cannot read (it plays Ironclad on v0.111.0 only).
    pub fn set_advisor(&mut self, advisor: &str) -> Result<(), SessionError> {
        match advisor {
            ADVISOR_SEARCH => {}
            ADVISOR_E2E if !cfg!(feature = "e2e") => {
                return Err(SessionError::new(400, "this build has no e2e feature"));
            }
            ADVISOR_E2E => {
                #[cfg(feature = "e2e")]
                self.e2e_check_run()?;
            }
            other => {
                return Err(SessionError::new(
                    400,
                    format!("advisor must be {ADVISOR_SEARCH}|{ADVISOR_E2E}, got '{other}'"),
                ));
            }
        }
        self.advisor = advisor.to_string();
        Ok(())
    }

    /// Whether this run was built as the e2e student's training run.
    #[must_use]
    pub fn e2e_setup(&self) -> bool {
        self.e2e_setup
    }

    /// Mark the run identity as the e2e start (set before `start_run_mode` /
    /// a restore, which then build through `Game::start_e2e`).
    pub fn set_e2e_setup(&mut self, on: bool) {
        self.e2e_setup = on;
    }

    /// The clock the e2e brain times its decisions with.
    pub fn set_perf_clock(&mut self, clock: Option<fn() -> f64>) {
        self.perf_clock = clock;
    }

    /// The live `Game`, if a run is active.
    #[must_use]
    pub fn game(&self) -> Option<&Game> {
        self.game.as_ref()
    }

    /// `_spawn_game`: a fresh driver opted into both display logs.
    pub(crate) fn spawn_game(&mut self) -> &mut Game {
        let mut game = Game::new();
        game.game_version = self.game_version;
        game.set_record_card_plays(true);
        game.set_record_combat_log(true);
        self.game = Some(game);
        self.game.as_mut().expect("just set")
    }

    pub(crate) fn require_game(&mut self) -> Result<&mut Game, SessionError> {
        self.game
            .as_mut()
            .ok_or_else(|| SessionError::new(400, "no live run"))
    }

    // ── display logs ───────────────────────────────────────────────────

    fn player_loc_id(&self) -> String {
        let c = if self.character.is_empty() {
            "PLAYER"
        } else {
            &self.character
        };
        c.to_uppercase().replace(' ', "_")
    }

    fn log_names(&self) -> LogNames {
        crate::enrich::log_names(self.loc)
    }

    /// Lengths of the two combat-scoped display logs, taken BEFORE an action.
    pub(crate) fn mark_display_logs(&self) -> (usize, usize) {
        match self.game.as_ref() {
            None => (0, 0),
            Some(g) => (g.card_play_log().len(), g.combat_log().len()),
        }
    }

    /// One action's tail of the two display logs, rendered for the UI. A log
    /// SHORTER than its mark means the action started a new fight — take it
    /// all.
    pub(crate) fn slice_display_logs(&self, marks: (usize, usize)) -> (Vec<String>, Vec<Value>) {
        let Some(game) = self.game.as_ref() else {
            return (Vec::new(), Vec::new());
        };
        let (play_before, clog_before) = marks;
        let mut play_lines = Vec::new();
        let full = game.card_play_log();
        if !full.is_empty() {
            let tail = if full.len() >= play_before {
                &full[play_before..]
            } else {
                &full[..]
            };
            let loc = self.loc;
            play_lines = format_play_log(tail, &|cid: &str| {
                resolve_name(Some(&format!("{cid}.title")), loc)
            });
        }
        let mut combat_rows = Vec::new();
        let full_log = game.combat_log();
        if !full_log.is_empty() {
            let tail = if full_log.len() >= clog_before {
                &full_log[clog_before..]
            } else {
                &full_log[..]
            };
            combat_rows = format_combat_log(
                tail,
                Some(&self.log_names()),
                false,
                VFX_ONLY_LOG_KINDS,
                &self.player_loc_id(),
            );
        }
        (play_lines, combat_rows)
    }

    // ── tree helpers ───────────────────────────────────────────────────

    // One argument per `Node` field the caller decides (the rest derive
    // from the parent) — a builder here would only restate the struct.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_node(
        &mut self,
        parent_id: Option<u64>,
        state: Value,
        action: Option<String>,
        args: Value,
        label: String,
        details: Vec<String>,
        combat_log: Vec<Value>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let idx = parent_id.map_or(0, |p| self.nodes[&p].idx + 1);
        let combat = self.game.as_ref().and_then(Game::combat_state).cloned();
        let decision = state
            .get("decision")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let value = self.value_for_new_node(&decision);
        self.nodes.insert(
            id,
            Node {
                id,
                parent_id,
                idx,
                decision,
                state,
                action,
                args,
                label,
                details,
                combat_log,
                error: false,
                child_ids: Vec::new(),
                value,
                combat,
            },
        );
        if let Some(p) = parent_id {
            self.nodes.get_mut(&p).expect("parent").child_ids.push(id);
        }
        id
    }

    /// `_value_for_new_node`: the value to stamp on a node being created —
    /// replayed off the restore queue when one is set (a save / persist
    /// restore already paid for them), else computed.
    fn value_for_new_node(&mut self, decision: &str) -> Option<f64> {
        if let Some(recorded) = self.restore_values.as_ref() {
            let i = self.restore_idx;
            self.restore_idx += 1;
            return recorded.get(i).copied().flatten();
        }
        self.node_value(decision)
    }

    /// `_node_value`: the PPO critic's V(s) of the freshly created node's
    /// state (the WPA review's data source), or `None` in combat, at game
    /// over, or when no net is loaded. Never disturbs play.
    fn node_value(&mut self, decision: &str) -> Option<f64> {
        if decision == "combat_play" || decision == "game_over" {
            return None;
        }
        let net_error = self.net_error();
        let rm = self.game.as_mut()?.native_run_mut()?;
        ppo::evaluate(self.ppo.as_ref(), &net_error, rm)
            .and_then(|ev| ev.value)
            .map(f64::from)
    }

    /// Make `root` the tree's root and active node (a fresh start).
    pub(crate) fn set_root(&mut self, root: u64) {
        self.root_id = Some(root);
        self.active_id = Some(root);
    }

    /// Nodes from the root to `node_id` inclusive.
    pub(crate) fn path_to(&self, node_id: u64) -> Vec<u64> {
        let mut chain = Vec::new();
        let mut cur = Some(node_id);
        while let Some(id) = cur {
            chain.push(id);
            cur = self.nodes[&id].parent_id;
        }
        chain.reverse();
        chain
    }

    pub(crate) fn active(&self) -> Result<&Node, SessionError> {
        let id = self
            .active_id
            .ok_or_else(|| SessionError::new(400, "no live run"))?;
        Ok(&self.nodes[&id])
    }

    /// The active node (400 when no run is live).
    pub fn active_node(&self) -> Result<&Node, SessionError> {
        self.active()
    }

    /// A node by id.
    #[must_use]
    pub fn node(&self, id: u64) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Root → active node, inclusive (400 when no run is live).
    pub fn active_path(&self) -> Result<Vec<&Node>, SessionError> {
        let active = self.active()?.id;
        Ok(self
            .path_to(active)
            .iter()
            .map(|id| &self.nodes[id])
            .collect())
    }

    /// The localization table the session renders with.
    #[must_use]
    pub fn loc(&self) -> &'static Loc {
        self.loc
    }

    /// `map_state` for a state about to become the active node's, which
    /// `e2e_reseed_map` finishes: under the e2e advisor that builds the map
    /// on the student's column, so building it here too would be wasted.
    fn map_state_live(&mut self, state: Value) -> Value {
        #[cfg(feature = "e2e")]
        if self.advisor_is_e2e() && !self.is_e2e_brain() {
            return state;
        }
        self.map_state(state)
    }

    /// `_attach_full_map`: at a `map_select` decision, attach the connected
    /// graph under `state["full_map"]` (best-effort), its route seeded on
    /// the driving brain.
    pub(crate) fn map_state(&mut self, mut state: Value) -> Value {
        if state.get("decision").and_then(Value::as_str) != Some("map_select") {
            return state;
        }
        let first_col = self.driving_first_col(Some(&state));
        let Some(game) = self.game.as_ref() else {
            return state;
        };
        let full_map = game.get_map(first_col);
        if full_map.as_object().is_some_and(|m| !m.is_empty()) {
            if let Some(m) = state.as_object_mut() {
                m.insert("full_map".into(), full_map);
            }
        }
        state
    }

    // ── lifecycle ──────────────────────────────────────────────────────

    /// Start the run, create the root node, return its state.
    pub fn start(
        &mut self,
        character: &str,
        seed: &str,
        ascension: u8,
        act1: &str,
    ) -> Result<Value, SessionError> {
        let seed = sts2_core::canonicalize_seed(seed);
        self.character = character.to_string();
        self.seed.clone_from(&seed);
        self.ascension = ascension;
        self.act1 = act1.to_string();
        let raw = self
            .spawn_game()
            .start(character, &seed, ascension, act1)
            .map_err(|e| SessionError::new(400, e.0))?;
        let state = self.map_state(raw);
        let label = format!("Started {character} (seed {seed}, A{ascension}, act1 {act1})");
        let root = self.add_node(None, state.clone(), None, json!({}), label, vec![], vec![]);
        self.root_id = Some(root);
        self.active_id = Some(root);
        Ok(state)
    }

    /// Rebuild a run from a pasted live-game position and make it the root.
    /// The graft's warnings (what could not be reproduced) are read back with
    /// [`Self::import_notes`]. The snapshot is retained so `seek`, a reload and
    /// a save reconstruct through the graft, not as a fresh seed run.
    pub fn start_imported(
        &mut self,
        snapshot: crate::state_import::RunSnapshot,
    ) -> Result<Value, SessionError> {
        self.character.clone_from(&snapshot.character);
        self.seed.clone_from(&snapshot.seed);
        self.ascension = snapshot.ascension;
        let raw = self
            .spawn_game()
            .start_imported(&snapshot)
            .map_err(|e| SessionError::new(400, e.0))?;
        // `act1` was resolved by the roll inside the graft; remember it so a
        // rebuild lands on the same maps.
        self.act1 = self
            .game
            .as_ref()
            .map_or("overgrowth", |g| g.act1.as_str())
            .to_string();
        let state = self.map_state(raw);
        let label = format!(
            "Imported {} (seed {}, A{}, act {}, floor {})",
            snapshot.character, snapshot.seed, snapshot.ascension, snapshot.act, snapshot.floor
        );
        let root = self.add_node(None, state.clone(), None, json!({}), label, vec![], vec![]);
        self.root_id = Some(root);
        self.active_id = Some(root);
        self.imported = Some(snapshot);
        Ok(state)
    }

    /// What the import graft could not reproduce (empty for a non-imported
    /// run). Surfaced next to the board so a half-restored state never reads
    /// as a whole one.
    #[must_use]
    pub fn import_notes(&self) -> Vec<String> {
        self.game
            .as_ref()
            .map(Game::import_notes)
            .unwrap_or_default()
    }

    /// Forward `action` to the game, append a child of the active node.
    /// Acting on a non-leaf forks a branch; the prior subtree is preserved.
    pub fn apply(&mut self, action: &str, args: &Value) -> Result<ApplyResult, SessionError> {
        if self.game.is_none() {
            return Err(SessionError::new(400, "no live run"));
        }
        if !ALLOWED_ACTIONS.contains(&action) {
            return Err(SessionError::new(
                400,
                format!("action not allowed: {action}"),
            ));
        }
        let parent = self.active()?;
        let parent_id = parent.id;
        // Resolve loc-key display names before rendering the label / diff so
        // the history reads "Play Strike", not "Play STRIKE_IRONCLAD.title".
        let parent_named = resolve_state_names(Some(&parent.state), self.loc);
        let label = describe_action(
            Some(&parent_named),
            Some(&json!({"cmd": "action", "action": action, "args": args})),
        );
        let marks = self.mark_display_logs();
        #[cfg(feature = "e2e")]
        let e2e_pre = self.e2e_pre_external();
        let resp = self.require_game()?.act(action, args);
        let resp = match resp {
            Ok(v) => v,
            Err(GameError(message)) => {
                // Illegal move: leave the tree untouched, re-sync the state.
                let raw = self
                    .require_game()?
                    .get_state()
                    .map_err(|e| SessionError::new(400, e.0))?;
                let state = self.map_state(raw);
                return Ok(ApplyResult {
                    state,
                    error: Some(message),
                    label,
                    details: Vec::new(),
                });
            }
        };
        let (play_lines, combat_rows) = self.slice_display_logs(marks);
        let resp_named = resolve_state_names(Some(&resp), self.loc);
        let mut details = play_lines;
        details.extend(diff_states(&parent_named, &resp_named));
        let state = self.map_state_live(resp);
        let node = self.add_node(
            Some(parent_id),
            state.clone(),
            Some(action.to_string()),
            args.clone(),
            label.clone(),
            details.clone(),
            combat_rows,
        );
        self.active_id = Some(node);
        #[cfg(feature = "e2e")]
        self.e2e_post_external(e2e_pre, node);
        #[cfg(feature = "e2e")]
        let state = {
            let mut state = state;
            self.e2e_reseed_map(&mut state);
            state
        };
        Ok(ApplyResult {
            state,
            error: None,
            label,
            details,
        })
    }

    /// Replay to `node_id` and make it the active node; return its state.
    /// Read-only: no new node is created.
    pub fn seek(&mut self, node_id: u64) -> Result<Value, SessionError> {
        if self.game.is_none() || self.root_id.is_none() {
            return Err(SessionError::new(400, "no live run"));
        }
        if !self.nodes.contains_key(&node_id) {
            return Err(SessionError::new(404, format!("unknown node: {node_id}")));
        }
        let path = self.path_to(node_id);
        if let Some(g) = self.game.as_mut() {
            g.close();
        }
        // Rebuild the run the SAME way it was originally constructed.
        let (character, seed, ascension, act1) = (
            self.character.clone(),
            self.seed.clone(),
            self.ascension,
            self.act1.clone(),
        );
        self.rebuild_game(&character, &seed, ascension, &act1)?;
        let mut last_resp: Option<Value> = None;
        for &id in &path[1..] {
            let (action, args) = {
                let n = &self.nodes[&id];
                (n.action.clone().unwrap_or_default(), n.args.clone())
            };
            last_resp = Some(self.replay_node(id, &action, &args)?);
        }
        self.active_id = Some(node_id);
        let state = match last_resp {
            Some(v) => self.map_state_live(v),
            None => self.nodes[&path[0]].state.clone(),
        };
        #[cfg(feature = "e2e")]
        let state = {
            let mut state = state;
            self.e2e_reseed_map(&mut state);
            state
        };
        self.nodes.get_mut(&node_id).expect("node").state = state.clone();
        Ok(state)
    }

    /// Re-apply one node's producing action during a replay: a native-agent
    /// node (`__agent_native__`) goes straight to `take_action`, anything
    /// else through `act`. Returns the raw post-action state; 409 names the
    /// node when the engine rejects the move.
    fn replay_node(&mut self, id: u64, action: &str, args: &Value) -> Result<Value, SessionError> {
        let game = self.require_game()?;
        if action == NATIVE_ACTION {
            if game.native_run().is_none() {
                return Err(SessionError::new(409, "no native run"));
            }
            let result = game.take_native_value(args).map_err(|e| {
                SessionError::new(409, format!("replay diverged at node {id}: {e}"))
            })?;
            if result != "advanced" && result != "run_over" {
                return Err(SessionError::new(
                    409,
                    format!("replay diverged at node {id}: take_action returned '{result}'"),
                ));
            }
            game.get_state().map_err(|e| SessionError::new(409, e.0))
        } else {
            game.act(action, args).map_err(|GameError(msg)| {
                SessionError::new(409, format!("replay diverged at node {id}: {msg}"))
            })
        }
    }

    /// `seek`'s constructor dispatch: a run-mode run is rebuilt through the
    /// faithful (integer-seed) constructor — or the e2e one for an e2e
    /// start — an interactive one through `start` — the two hash the seed
    /// differently.
    fn rebuild_game(
        &mut self,
        character: &str,
        seed: &str,
        ascension: u8,
        act1: &str,
    ) -> Result<(), SessionError> {
        let mut game = self.spawn_game_detached();
        self.start_identity(&mut game, character, seed, ascension, act1)?;
        self.game = Some(game);
        Ok(())
    }

    /// A fresh driver opted into both display logs, not installed.
    fn spawn_game_detached(&self) -> Game {
        let mut game = Game::new();
        game.game_version = self.game_version;
        game.set_record_card_plays(true);
        game.set_record_combat_log(true);
        game
    }

    /// The run as this session constructed it, in a NEW game without display
    /// logs (the e2e brain's shadow replay).
    #[cfg(feature = "e2e")]
    pub(crate) fn construct_game(&self) -> Result<Game, SessionError> {
        let mut game = Game::new();
        game.game_version = self.game_version;
        let (character, seed, act1) =
            (self.character.clone(), self.seed.clone(), self.act1.clone());
        self.start_identity(&mut game, &character, &seed, self.ascension, &act1)?;
        Ok(game)
    }

    fn start_identity(
        &self,
        game: &mut Game,
        character: &str,
        seed: &str,
        ascension: u8,
        act1: &str,
    ) -> Result<(), SessionError> {
        if let Some(snapshot) = self.imported.as_ref() {
            // An imported run is grafted, not seed-derived: rebuild it the same
            // way it was first constructed, or the deck / cursors diverge.
            game.start_imported(snapshot)
                .map_err(|e| SessionError::new(409, e.0))?;
        } else if self.run_mode_run {
            let seed_int = seed.parse::<i32>().map_err(|_| {
                SessionError::new(
                    409,
                    format!("run-mode run carries a non-numeric seed {seed:?}; cannot rebuild it"),
                )
            })?;
            if self.e2e_setup {
                start_e2e_game(game, seed_int, ascension)?;
            } else {
                game.start_faithful(character, seed_int, ascension)
                    .map_err(|e| SessionError::new(409, e.0))?;
            }
        } else {
            game.start(character, seed, ascension, act1)
                .map_err(|e| SessionError::new(409, e.0))?;
        }
        Ok(())
    }

    /// The active node's current state (re-synced from the game).
    pub fn state(&mut self) -> Result<Value, SessionError> {
        let raw = self
            .require_game()?
            .get_state()
            .map_err(|e| SessionError::new(400, e.0))?;
        let state = self.map_state_live(raw);
        #[cfg(feature = "e2e")]
        let state = {
            let mut state = state;
            self.e2e_reseed_map(&mut state);
            state
        };
        let id = self.active()?.id;
        self.nodes.get_mut(&id).expect("node").state = state.clone();
        Ok(state)
    }

    /// The current act's full connected map graph, on demand, or `None`
    /// when no map is available (e.g. the boss room).
    pub fn map(&mut self) -> Result<Option<Value>, SessionError> {
        let first_col = self.driving_first_col(None);
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| SessionError::new(400, "no live run"))?;
        let full_map = game.get_map(first_col);
        if full_map.as_object().is_some_and(|m| !m.is_empty()) {
            Ok(Some(full_map))
        } else {
            Ok(None)
        }
    }

    /// The branch tree metadata plus the active path (root → leaf), each
    /// node carrying the `floor` / `act` it sits on and its combat `turn`.
    #[must_use]
    pub fn timeline(&self) -> Value {
        let Some(active) = self.active_id else {
            return json!({"nodes": [], "active_path": [], "active_leaf": Value::Null});
        };
        let mut resolved: BTreeMap<u64, (Value, Value)> = BTreeMap::new();
        let mut nodes = Vec::with_capacity(self.nodes.len());
        for node in self.nodes.values() {
            let (pf, pa) = node
                .parent_id
                .and_then(|p| resolved.get(&p).cloned())
                .unwrap_or((Value::Null, Value::Null));
            let floor = node.state.get("floor").cloned().unwrap_or(Value::Null);
            let act = node.state.get("act").cloned().unwrap_or(Value::Null);
            let cur = (
                if floor.is_null() { pf } else { floor },
                if act.is_null() { pa } else { act },
            );
            resolved.insert(node.id, cur.clone());
            let mut meta = node.meta();
            let m = meta.as_object_mut().expect("meta dict");
            m.insert("floor".into(), cur.0);
            m.insert("act".into(), cur.1);
            m.insert(
                "turn".into(),
                node.state.get("round").cloned().unwrap_or(Value::Null),
            );
            nodes.push(meta);
        }
        json!({
            "nodes": nodes,
            "active_path": self.path_to(active),
            "active_leaf": active,
        })
    }

    // ── named saves (active line only) ─────────────────────────────────

    fn progress(&self, path: &[u64]) -> Progress {
        let mut floor = Value::Null;
        let mut act = Value::Null;
        for id in path {
            let st = &self.nodes[id].state;
            if let Some(f) = st.get("floor").filter(|v| !v.is_null()) {
                floor = f.clone();
            }
            if let Some(a) = st.get("act").filter(|v| !v.is_null()) {
                act = a.clone();
            }
        }
        let leaf = &self.nodes[path.last().expect("non-empty path")].state;
        let player = leaf.get("player").cloned().unwrap_or(Value::Null);
        Progress {
            act,
            floor,
            hp: player.get("hp").cloned().unwrap_or(Value::Null),
            max_hp: player.get("max_hp").cloned().unwrap_or(Value::Null),
        }
    }

    /// `_progress` over an already-resolved path of nodes.
    pub(crate) fn progress_of_path(&self, path: &[&Node]) -> Progress {
        let ids: Vec<u64> = path.iter().map(|n| n.id).collect();
        self.progress(&ids)
    }

    /// `(line, progress)` for the active path: the ordered `{action, args}`
    /// pairs producing the leaf (root excluded) plus leaf context.
    pub fn active_line(&self) -> Result<(Vec<Value>, Progress), SessionError> {
        let active = self.active()?.id;
        let path = self.path_to(active);
        let line = path[1..]
            .iter()
            .map(|id| {
                let n = &self.nodes[id];
                json!({"action": n.action, "args": n.args})
            })
            .collect();
        Ok((line, self.progress(&path)))
    }

    /// PPO critic values along the active path, root first (`active_node_values`).
    pub fn active_node_values(&self) -> Result<Vec<Option<f64>>, SessionError> {
        let active = self.active()?.id;
        Ok(self
            .path_to(active)
            .iter()
            .map(|id| self.nodes[id].value)
            .collect())
    }

    /// Start a fresh run and replay `line`, rebuilding the branch tree.
    /// Returns the active leaf state; 409 if any replayed action is rejected.
    ///
    /// `run_mode` MUST match how the saved run was started (a run-mode line
    /// replays against the integer-seed faithful run; an interactive one
    /// against the text-seed run), and `act1` how it was recorded (GH #544).
    /// `node_values` are the saving session's `active_node_values`, replayed
    /// onto the rebuilt nodes instead of re-running the critic.
    #[allow(clippy::too_many_arguments)]
    pub fn load_line(
        &mut self,
        character: &str,
        seed: &str,
        ascension: u8,
        line: &[Value],
        act1: &str,
        run_mode: bool,
        node_values: Option<Vec<Option<f64>>>,
    ) -> Result<Value, SessionError> {
        self.restore_values = node_values;
        self.restore_idx = 0;
        let out = self.load_line_inner(character, seed, ascension, line, act1, run_mode);
        // Only the restore replays values; everything after it — including
        // a fork off a restored node — must compute its own.
        self.restore_values = None;
        out
    }

    fn load_line_inner(
        &mut self,
        character: &str,
        seed: &str,
        ascension: u8,
        line: &[Value],
        act1: &str,
        run_mode: bool,
    ) -> Result<Value, SessionError> {
        let mut state = if run_mode {
            let seed_int = seed.parse::<i32>().map_err(|_| {
                SessionError::new(
                    409,
                    format!("run-mode save carries a non-numeric seed {seed:?}; cannot restore it"),
                )
            })?;
            // Benchmark, matching `Session::new`'s defaults — a restored
            // run-mode line must not resume at a fifth of the budget it was
            // recorded at just because the restore path hard-coded Quick.
            let brain = if self.e2e_setup { "e2e" } else { "ppo" };
            self.start_run_mode(
                character, seed_int, ascension, brain, "rust", 1000, 4000, 1, None,
            )?
        } else {
            self.start(character, seed, ascension, act1)?
        };
        let total = line.len();
        for (i, entry) in line.iter().enumerate() {
            let action = entry
                .get("action")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let args = entry
                .get("args")
                .cloned()
                .filter(Value::is_object)
                .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
            let result = if action == NATIVE_ACTION {
                self.apply_native(&args)?
            } else {
                self.apply(&action, &args)?
            };
            if let Some(err) = result.error {
                let at = result
                    .state
                    .get("decision")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                return Err(SessionError::new(
                    409,
                    format!(
                        "replay diverged at step {}/{total} (action '{action}'): {err}; \
                         engine is at '{at}'",
                        i + 1
                    ),
                ));
            }
            state = result.state;
        }
        Ok(state)
    }

    /// Tear the session down WITHOUT recording anything.
    pub fn discard(&mut self) {
        if let Some(g) = self.game.as_mut() {
            g.close();
        }
        self.game = None;
        self.active_id = None;
    }

    /// End the run. `log_path` names the JSONL run log the caller wrote
    /// (`runlog::write_run_log`) — the host keeps the text; Python answered
    /// its file path here.
    pub fn quit(&mut self, log_path: Option<String>) -> Value {
        self.discard();
        json!({"ok": true, "log_path": log_path})
    }

    // ── persistence of the whole tree (browser reload / process death) ──

    /// Everything needed to rebuild this session without node states: the
    /// identity, every node's producing action, and the active leaf.
    /// Restore replays the active path (states materialize); cold branches
    /// materialize on `seek`.
    #[must_use]
    pub fn export_persist(&self) -> Value {
        let nodes: Vec<Value> = self
            .nodes
            .values()
            .map(|n| {
                json!({
                    "id": n.id,
                    "parent_id": n.parent_id,
                    "action": n.action,
                    "args": n.args,
                    "label": n.label,
                    "details": n.details,
                    "combat_log": n.combat_log,
                    "error": n.error,
                    "decision": n.decision,
                    "floor": n.state.get("floor").cloned().unwrap_or(Value::Null),
                    "act": n.state.get("act").cloned().unwrap_or(Value::Null),
                    "round": n.state.get("round").cloned().unwrap_or(Value::Null),
                    "value": n.value,
                })
            })
            .collect();
        let mut out = json!({
            "schema_version": 1,
            "character": self.character,
            "seed": self.seed,
            "ascension": self.ascension,
            "act1": self.act1,
            "backend": self.backend(),
            "run_mode": self.run_mode(),
            "nodes": nodes,
            "active_leaf": self.active_id,
            "faithful_steps": self.faithful_steps,
            "run_mode_config": self.run_mode_config(),
            // Carried so a reload rebuilds an imported run through the graft;
            // `null` for a seed-derived run.
            "imported": self
                .imported
                .as_ref()
                .and_then(|s| serde_json::to_value(s).ok())
                .unwrap_or(Value::Null),
        });
        // GH #801: named only when not the default, so a v0.107.1 persist
        // keeps its bytes; a reader treats absence as v0.107.1.
        if !self.game_version.is_default() {
            out["game_version"] = json!(self.game_version.as_str());
        }
        // Named only for an e2e start (every other persist keeps its bytes).
        if self.e2e_setup {
            out["e2e_setup"] = json!(true);
        }
        out
    }

    /// Rebuild from [`Self::export_persist`] output into a session already
    /// built in the payload's `game_version` ([`game_version_of`]).
    /// `progress` is called with `(done, total)` as the active path replays.
    /// Returns the active leaf state.
    pub fn import_persist(
        &mut self,
        payload: &Value,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Value, SessionError> {
        let str_of = |k: &str| {
            payload
                .get(k)
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        };
        let character = str_of("character");
        let seed = str_of("seed");
        // The build is the caller's: `Workbench::import_persist` parses the
        // payload's `game_version` (and owns its 409) and builds this session
        // in it, so it is not parsed a second time here (#855).
        debug_assert_eq!(game_version_of(payload).ok(), Some(self.game_version));
        let act1 = {
            let a = str_of("act1");
            if a.is_empty() {
                "overgrowth".to_string()
            } else {
                a
            }
        };
        let ascension = u8::try_from(
            payload
                .get("ascension")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        )
        .unwrap_or(u8::MAX);
        let empty = Vec::new();
        let nodes = payload
            .get("nodes")
            .and_then(Value::as_array)
            .unwrap_or(&empty);
        let active_leaf = payload.get("active_leaf").and_then(Value::as_u64);
        if character.is_empty() || nodes.is_empty() {
            return Err(SessionError::new(409, "persisted session is empty"));
        }
        let run_mode = payload
            .get("run_mode")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        self.e2e_setup = run_mode
            && payload
                .get("e2e_setup")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        // Re-create every node cold, in id order, so ids and parents match the
        // persisted tree. Node 0 is the root `start` creates.
        let mut order: Vec<&Value> = nodes.iter().collect();
        order.sort_by_key(|n| n.get("id").and_then(Value::as_u64).unwrap_or(0));
        // The persisted values are replayed onto the cold nodes (creation
        // order = id order); the critic must not score a cold node's state.
        self.restore_values = Some(
            order
                .iter()
                .map(|n| n.get("value").and_then(Value::as_f64))
                .collect(),
        );
        self.restore_idx = 0;
        // An imported run rebuilds through the graft, not from the seed.
        let imported_snapshot: Option<crate::state_import::RunSnapshot> = payload
            .get("imported")
            .filter(|v| !v.is_null())
            .and_then(|v| serde_json::from_value(v.clone()).ok());
        let started = if let Some(snapshot) = imported_snapshot {
            self.start_imported(snapshot)
        } else if run_mode {
            let cfg = payload
                .get("run_mode_config")
                .cloned()
                .unwrap_or(Value::Null);
            let get_u32 = |k: &str, d: u32| {
                cfg.get(k)
                    .and_then(Value::as_u64)
                    .and_then(|v| u32::try_from(v).ok())
                    .filter(|v| *v > 0)
                    .unwrap_or(d)
            };
            let brain = cfg
                .get("off_combat_brain")
                .and_then(Value::as_str)
                .unwrap_or("ppo")
                .to_string();
            let ppo_impl = cfg
                .get("ppo_impl")
                .and_then(Value::as_str)
                .unwrap_or("rust")
                .to_string();
            match seed.parse::<i32>() {
                Ok(seed_int) => self.start_run_mode(
                    &character,
                    seed_int,
                    ascension,
                    &brain,
                    &ppo_impl,
                    get_u32("combat_sims", 1000),
                    get_u32("combat_boss_sims", 4000),
                    get_u32("combat_threads", 1),
                    cfg.get("reward_sims").and_then(Value::as_i64),
                ),
                Err(_) => Err(SessionError::new(
                    409,
                    format!("persisted run-mode session carries a non-numeric seed {seed:?}"),
                )),
            }
        } else {
            self.start(&character, &seed, ascension, &act1)
        };
        let mut state = match started {
            Ok(v) => v,
            Err(e) => {
                self.restore_values = None;
                return Err(e);
            }
        };
        for n in order.iter().skip(1) {
            let parent_id = n.get("parent_id").and_then(Value::as_u64);
            let id = self.add_node(
                parent_id,
                json!({
                    "decision": n.get("decision").cloned().unwrap_or(json!("")),
                    "floor": n.get("floor").cloned().unwrap_or(Value::Null),
                    "act": n.get("act").cloned().unwrap_or(Value::Null),
                    "round": n.get("round").cloned().unwrap_or(Value::Null),
                }),
                n.get("action").and_then(Value::as_str).map(str::to_string),
                n.get("args").cloned().unwrap_or_else(|| json!({})),
                n.get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                n.get("details")
                    .and_then(Value::as_array)
                    .map(|d| {
                        d.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default(),
                n.get("combat_log")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
            );
            if let Some(node) = self.nodes.get_mut(&id) {
                node.error = n.get("error").and_then(Value::as_bool).unwrap_or(false);
            }
        }
        self.restore_values = None;
        // The advisor rides the run-mode config of every run (absent = search);
        // a run the student cannot read falls back to search.
        if let Some(a) = payload
            .pointer("/run_mode_config/advisor")
            .and_then(Value::as_str)
        {
            if self.set_advisor(a).is_err() {
                self.advisor = ADVISOR_SEARCH.to_string();
            }
        }
        self.faithful_steps = payload
            .get("faithful_steps")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        // Replay the active path to materialize its states.
        let leaf = active_leaf.unwrap_or(0);
        if !self.nodes.contains_key(&leaf) {
            return Err(SessionError::new(
                409,
                format!("persisted active leaf {leaf} is unknown"),
            ));
        }
        let path = self.path_to(leaf);
        let total = path.len().saturating_sub(1);
        progress(0, total);
        for (done, &id) in path.iter().enumerate().skip(1) {
            let (action, args) = {
                let n = &self.nodes[&id];
                (n.action.clone().unwrap_or_default(), n.args.clone())
            };
            let v = self.replay_node(id, &action, &args)?;
            let st = self.map_state(v);
            self.nodes.get_mut(&id).expect("node").state = st.clone();
            state = st;
            progress(done, total);
        }
        self.active_id = Some(leaf);
        Ok(state)
    }

    /// The node's stored args as a JSON object (for tests / the host).
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Active leaf id.
    #[must_use]
    pub fn active_leaf(&self) -> Option<u64> {
        self.active_id
    }
}

/// The `game_version` a stored payload (save, persist, run-log meta)
/// carries: absent or `null` is the default build (every payload written
/// before GH #801, and every default-build one since). An unknown string is
/// an error naming the supported builds.
pub fn game_version_of(payload: &Value) -> Result<GameVersion, String> {
    parse_game_version(payload.get("game_version"))
}

/// A `game_version` value as the wire carries it (see [`game_version_of`]):
/// `None` / `null` is the default build; otherwise exactly one of the
/// supported spellings (`"v0.107.1"`, `"v0.111.0"`).
pub fn parse_game_version(value: Option<&Value>) -> Result<GameVersion, String> {
    match value {
        None | Some(Value::Null) => Ok(GameVersion::default()),
        Some(Value::String(s)) => GameVersion::ALL
            .into_iter()
            .find(|v| v.as_str() == s)
            .ok_or_else(|| {
                format!(
                    "unknown game_version {s:?} (supported: {})",
                    supported_game_versions()
                )
            }),
        Some(other) => Err(format!(
            "game_version must be a string (supported: {}), got {other}",
            supported_game_versions()
        )),
    }
}

/// `"v0.107.1, v0.111.0"` — the supported builds, for error messages.
#[must_use]
pub fn supported_game_versions() -> String {
    GameVersion::ALL
        .iter()
        .map(|v| v.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Build a `{"action", "args"}` line entry.
#[must_use]
pub fn line_entry(action: &str, args: Value) -> Value {
    let mut m = Map::new();
    m.insert("action".into(), json!(action));
    m.insert("args".into(), args);
    Value::Object(m)
}

/// [`Game::start_e2e`], or a 409 when this build has no e2e feature.
fn start_e2e_game(game: &mut Game, seed: i32, ascension: u8) -> Result<Value, SessionError> {
    #[cfg(feature = "e2e")]
    {
        game.start_e2e(seed, ascension)
            .map_err(|e| SessionError::new(409, e.0))
    }
    #[cfg(not(feature = "e2e"))]
    {
        let _ = (game, seed, ascension);
        Err(SessionError::new(
            409,
            "this build has no e2e feature; cannot build an e2e run",
        ))
    }
}
