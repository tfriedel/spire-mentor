//! Run mode on the [`Session`] (M3): the orchestration half of
//! `BranchingLiveSession` — `start_run_mode`, `run_mode_step` /
//! `play_to_end`, the autobattle loop and verdict, the two-axis
//! `set_run_mode`, "play best" with the PPO brain, the PPO panel, and the
//! WPA review / verify. The pure cores live in [`crate::runmode`] and
//! [`crate::ppo`].

use serde_json::{json, Value};
use sts2_core::run_manager::RunAction;
use sts2_wire::emit;

use crate::advisors::{best_action, native_action};
use crate::game::GameError;
use crate::localization::resolve_state_names;
use crate::options::normalize_options;
use crate::ppo;
use crate::render::{describe_action, NATIVE_ACTION};
use crate::rmx;
use crate::runmode::{self, RunAgent};
use crate::session::{ApplyResult, Session, SessionError};
use crate::statediff::diff_states;

/// A millisecond clock for the `wpa-verify` deadline. `None` = no clock
/// (the arm is bounded by its step cap only).
pub type Clock<'a> = Option<&'a dyn Fn() -> f64>;

/// The brains `set_run_mode` accepts: `e2e` (the e2e Ironclad student
/// playing every decision, combat included) only in builds with the `e2e`
/// feature.
#[cfg(feature = "e2e")]
const BRAINS: &[&str] = &["heuristic", "ppo", "e2e"];
#[cfg(not(feature = "e2e"))]
const BRAINS: &[&str] = &["heuristic", "ppo"];

/// `"no faithful run; call start_faithful() first"`.
fn no_faithful_run() -> SessionError {
    SessionError::new(400, "no faithful run; call start_faithful() first")
}

fn action_label(state_named: &Value, verb: &str, args: &Value) -> String {
    describe_action(
        Some(state_named),
        Some(&json!({"cmd": "action", "action": verb, "args": args})),
    )
}

impl Session {
    /// The resolved two-axis run mode (`POST /api/live/run-mode`'s reply).
    #[must_use]
    pub fn run_mode_config(&self) -> Value {
        let mut out = json!({
            "off_combat_brain": self.off_combat_brain,
            "ppo_impl": self.ppo_impl,
            "combat_sims": self.combat_sims,
            "combat_boss_sims": self.combat_boss_sims,
            "combat_threads": self.combat_threads,
            "reward_sims": self.reward_sims,
        });
        // Named only when not the default, so every search-advised reply and
        // persist keeps its bytes (the golden parity hashes them); a reader
        // treats absence as `search`.
        if self.advisor_is_e2e() {
            out["advisor"] = json!(self.advisor);
        }
        out
    }

    /// `set_run_mode`: update any subset of the two-axis run mode. 400
    /// (Python's `ValueError`) on a bad value. A changed combat budget is
    /// pushed straight onto a live agent, RNG and decision counter intact.
    #[allow(clippy::too_many_arguments)]
    pub fn set_run_mode(
        &mut self,
        off_combat_brain: Option<&str>,
        ppo_impl: Option<&str>,
        combat_sims: Option<i64>,
        combat_boss_sims: Option<i64>,
        combat_threads: Option<i64>,
        reward_sims: Option<i64>,
    ) -> Result<(), SessionError> {
        if let Some(b) = off_combat_brain {
            if !BRAINS.contains(&b) {
                return Err(SessionError::new(
                    400,
                    format!("off_combat_brain must be {}, got '{b}'", BRAINS.join("|")),
                ));
            }
            self.off_combat_brain = b.to_string();
        }
        if let Some(i) = ppo_impl {
            if i != "python" && i != "rust" {
                return Err(SessionError::new(
                    400,
                    format!("ppo_impl must be python|rust, got '{i}'"),
                ));
            }
            self.ppo_impl = i.to_string();
        }
        let positive = |name: &str, val: Option<i64>| -> Result<Option<u32>, SessionError> {
            match val {
                None => Ok(None),
                Some(v) if v <= 0 => Err(SessionError::new(
                    400,
                    format!("{name} must be positive, got {v}"),
                )),
                Some(v) => Ok(Some(u32::try_from(v).unwrap_or(u32::MAX))),
            }
        };
        let sims = positive("combat_sims", combat_sims)?;
        let boss = positive("combat_boss_sims", combat_boss_sims)?;
        let threads = positive("combat_threads", combat_threads)?;
        if let Some(v) = sims {
            self.combat_sims = v;
        }
        if let Some(v) = boss {
            self.combat_boss_sims = v;
        }
        if let Some(v) = threads {
            self.combat_threads = v;
        }
        if let Some(r) = reward_sims {
            if r < 0 {
                return Err(SessionError::new(
                    400,
                    format!("reward_sims must be >= 0, got {r}"),
                ));
            }
            self.reward_sims = Some(r);
        }
        if let Some(agent) = self.agent.as_mut() {
            if sims.is_some() || boss.is_some() {
                agent.set_budget(sims, boss);
            }
        }
        Ok(())
    }

    /// `start_run_mode`: a parity-pinned run for run-mode autoplay — built
    /// the way `run_arena` / `PpoRunEnv` build theirs, with the native
    /// combat agent constructed once at the given budget.
    #[allow(clippy::too_many_arguments)]
    pub fn start_run_mode(
        &mut self,
        character: &str,
        seed: i32,
        ascension: u8,
        off_combat_brain: &str,
        ppo_impl: &str,
        combat_sims: u32,
        combat_boss_sims: u32,
        threads: u32,
        reward_sims: Option<i64>,
    ) -> Result<Value, SessionError> {
        self.set_run_mode(
            Some(off_combat_brain),
            Some(ppo_impl),
            Some(i64::from(combat_sims)),
            Some(i64::from(combat_boss_sims)),
            Some(i64::from(threads)),
            reward_sims,
        )?;
        self.run_mode_run = true;
        self.character = character.to_string();
        self.seed = seed.to_string();
        self.ascension = ascension;
        self.act1 = "overgrowth".to_string();
        let raw = if self.e2e_setup {
            // The e2e student's training run (`Game::start_e2e`).
            self.act1 = "roll".to_string();
            self.start_e2e_raw(seed, ascension)?
        } else {
            self.spawn_game()
                .start_faithful(character, seed, ascension)
                .map_err(|e| SessionError::new(400, e.0))?
        };
        let state = self.map_state(raw);
        let label = format!("Started {character} (seed {seed}, A{ascension}, {off_combat_brain})");
        let root = self.add_node(None, state.clone(), None, json!({}), label, vec![], vec![]);
        self.set_root(root);
        self.autobattle_prev_state = None;
        self.autobattle_pending = None;
        self.autobattle_autoplayed = 0;
        self.agent = Some(RunAgent::new(seed, combat_sims, combat_boss_sims));
        self.faithful_steps = 0;
        #[cfg(feature = "e2e")]
        if self.is_e2e_brain() {
            self.e2e_seed_root();
        }
        Ok(state)
    }

    /// `Game::start_e2e` on a fresh game (400 in a build without `e2e`).
    fn start_e2e_raw(&mut self, seed: i32, ascension: u8) -> Result<Value, SessionError> {
        #[cfg(feature = "e2e")]
        {
            self.spawn_game()
                .start_e2e(seed, ascension)
                .map_err(|e| SessionError::new(400, e.0))
        }
        #[cfg(not(feature = "e2e"))]
        {
            let _ = (seed, ascension);
            Err(SessionError::new(400, "this build has no e2e feature"))
        }
    }

    /// `_apply_native`: replay one recorded native-agent action, appending
    /// a branch node — the `apply` counterpart for `__agent_native__` nodes.
    pub fn apply_native(&mut self, args: &Value) -> Result<ApplyResult, SessionError> {
        if self.game.is_none() {
            return Err(SessionError::new(400, "no live run"));
        }
        let parent = self.active()?;
        let parent_id = parent.id;
        let parent_named = resolve_state_names(Some(&parent.state), self.loc);
        let label = {
            let game = self.game.as_ref().expect("checked");
            match game.native_run() {
                Some(rm) => {
                    let (verb, wb_args) = runmode::to_workbench_verb(rm, args);
                    action_label(&parent_named, &verb, &wb_args)
                }
                None => args
                    .get("action")
                    .map_or_else(|| "?".to_string(), crate::render::py::to_str),
            }
        };
        let marks = self.mark_display_logs();
        if self.game.as_ref().and_then(|g| g.native_run()).is_none() {
            return Err(SessionError::new(400, "no live run"));
        }
        #[cfg(feature = "e2e")]
        let e2e_pre = self.e2e_pre_external();
        let result = match self.require_game()?.take_native_value(args) {
            Ok(r) => r,
            Err(e) => {
                let state = self.state()?;
                return Ok(ApplyResult {
                    state,
                    error: Some(format!("take_action returned '{e}'")),
                    label,
                    details: Vec::new(),
                });
            }
        };
        if result != "advanced" && result != "run_over" {
            let state = self.state()?;
            return Ok(ApplyResult {
                state,
                error: Some(format!("take_action returned '{result}'")),
                label,
                details: Vec::new(),
            });
        }
        let raw = self
            .require_game()?
            .get_state()
            .map_err(|e| SessionError::new(400, e.0))?;
        let (play_lines, combat_rows) = self.slice_display_logs(marks);
        let mut details = play_lines;
        details.extend(diff_states(
            &parent_named,
            &resolve_state_names(Some(&raw), self.loc),
        ));
        let state = self.map_state(raw);
        let node = self.add_node(
            Some(parent_id),
            state.clone(),
            Some(NATIVE_ACTION.to_string()),
            args.clone(),
            label.clone(),
            details.clone(),
            combat_rows,
        );
        self.active_id = Some(node);
        #[cfg(feature = "e2e")]
        self.e2e_post_external(e2e_pre, node);
        Ok(ApplyResult {
            state,
            error: None,
            label,
            details,
        })
    }

    /// `_agent_step`: apply ONE native-agent decision and record a node.
    fn agent_step(&mut self) -> Result<Value, SessionError> {
        if self.game.is_none() || self.agent.is_none() {
            return Err(no_faithful_run());
        }
        let parent = self.active()?;
        let parent_id = parent.id;
        let parent_named = resolve_state_names(Some(&parent.state), self.loc);
        let marks = self.mark_display_logs();
        let rec = {
            let game = self.game.as_mut().expect("checked");
            if game.native_run().is_none() {
                return Err(SessionError::new(400, "no live run"));
            }
            self.agent.as_mut().expect("checked").step(game)
        };
        let Some(rec) = rec else {
            // The policy sees no legal actions — the run is finished.
            let state = self.state()?;
            let over = state.get("decision").and_then(Value::as_str) == Some("game_over")
                || self.native_over();
            let won = over && self.native_won();
            return Ok(json!({
                "state": state,
                "steps": self.faithful_steps,
                "victory": won,
                "won": won,
                "over": over,
                "error": Value::Null,
                "done": true,
                "debug": Value::Null,
                "search": Value::Null,
            }));
        };
        let raw = self
            .require_game()?
            .get_state()
            .map_err(|e| SessionError::new(400, e.0))?;
        let (play_lines, combat_rows) = self.slice_display_logs(marks);
        let mut details = play_lines;
        details.extend(diff_states(
            &parent_named,
            &resolve_state_names(Some(&raw), self.loc),
        ));
        let mut state = self.map_state(raw);
        if let Some(cs) = rec.combat_search.as_ref() {
            state["mcts"] = cs.clone();
        }
        let native = rmx::action_value(&rec.action);
        let verb = native
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();
        let act_label = {
            let rm = self
                .game
                .as_ref()
                .and_then(|g| g.native_run())
                .ok_or_else(|| SessionError::new(400, "no live run"))?;
            let (wb_verb, wb_args) = runmode::to_workbench_verb(rm, &native);
            action_label(&parent_named, &wb_verb, &wb_args)
        };
        let search = rec.combat_search.as_ref().map(|cs| {
            json!({
                "scores": crate::advisors::rows_from_mcts(cs, &parent_named),
                "action": act_label,
                "enemies": parent_named.get("enemies").cloned().unwrap_or_else(|| json!([])),
            })
        });
        let node = self.add_node(
            Some(parent_id),
            state.clone(),
            Some(NATIVE_ACTION.to_string()),
            native,
            act_label.clone(),
            details,
            combat_rows,
        );
        self.active_id = Some(node);
        self.faithful_steps += 1;
        let mut debug = json!({
            "phase": rec.phase,
            "is_combat": rec.is_combat,
            "action": act_label,
            "result": rec.result,
        });
        if let Some(cs) = rec.combat_search.as_ref() {
            debug["combat_search"] = cs.clone();
            debug["summary"] = json!(format!("combat MCTS → {act_label}"));
        } else {
            debug["summary"] = json!(format!("{} → {act_label}", rec.phase));
        }
        let _ = verb;
        Ok(json!({
            "state": state,
            "steps": self.faithful_steps,
            "victory": rec.won,
            "won": rec.won,
            "over": rec.over,
            "error": Value::Null,
            "done": false,
            "debug": debug,
            "search": search,
        }))
    }

    fn native_over(&self) -> bool {
        self.game
            .as_ref()
            .and_then(|g| g.native_run())
            .is_some_and(sts2_core::run_manager::RunManager::is_over)
    }

    fn native_won(&self) -> bool {
        self.game
            .as_ref()
            .and_then(|g| g.native_run())
            .is_some_and(|rm| rm.run_state.player_won)
    }

    fn native_phase_is_combat(&self) -> bool {
        self.game
            .as_ref()
            .and_then(|g| g.native_run())
            .is_some_and(|rm| rmx::phase_str(rm) == "combat")
    }

    /// `_brain_recommend`: the `ppo` brain's off-combat pick, degrading to
    /// the heuristic when the net can't decide.
    fn brain_recommend(&mut self) -> Option<Value> {
        #[cfg(feature = "e2e")]
        if self.is_e2e_brain() {
            if let Some(ev) = self.e2e_evaluate() {
                return Some(ev.pick);
            }
        }
        self.ppo_recommend().or_else(|| self.heuristic_recommend())
    }

    /// `run_mode_step`: apply ONE decision under the current run mode and
    /// record a node. Combat and the `heuristic` brain go through the
    /// native agent; the `ppo` brain resolves off-combat decisions itself
    /// and applies them through `apply`.
    pub fn run_mode_step(&mut self) -> Result<Value, SessionError> {
        if self.game.is_none() {
            return Err(SessionError::new(400, "no live run"));
        }
        if self.game.as_ref().and_then(|g| g.native_run()).is_none() {
            return Err(SessionError::new(400, "no live run"));
        }
        if self.agent.is_none() {
            // A hand-played run has no combat agent yet: build it lazily from
            // the session's budget so autoplay works on ANY live run.
            let seed = runmode::hash_seed(&self.seed);
            self.agent = Some(RunAgent::new(seed, self.combat_sims, self.combat_boss_sims));
        }
        #[cfg(feature = "e2e")]
        if self.e2e_drives() {
            return self.e2e_step();
        }
        if self.off_combat_brain == "heuristic" || self.native_phase_is_combat() {
            return self.agent_step();
        }
        let Some(rec) = self.brain_recommend() else {
            return self.agent_step();
        };
        let action = rec["action"].as_str().unwrap_or("").to_string();
        let args = rec.get("args").cloned().unwrap_or_else(|| json!({}));
        let result = self.apply(&action, &args)?;
        let over = result.state.get("decision").and_then(Value::as_str) == Some("game_over")
            || self.native_over();
        let won = over && self.native_won();
        if result.error.is_none() {
            self.faithful_steps += 1;
        }
        Ok(json!({
            "state": result.state,
            "steps": self.faithful_steps,
            "victory": won,
            "won": won,
            "over": over,
            "error": result.error,
            "done": over || result.error.is_some(),
            "debug": {"summary": result.label, "is_combat": false},
            "search": Value::Null,
        }))
    }

    /// `play_to_end`: a loop over [`Self::run_mode_step`]; `steps` counts
    /// the actions applied by THIS call.
    pub fn play_to_end(&mut self, max_steps: u64) -> Result<Value, SessionError> {
        if self.game.is_none() || self.agent.is_none() {
            return Err(no_faithful_run());
        }
        let before = self.faithful_steps;
        let mut out = json!({
            "state": self.active()?.state,
            "victory": false,
            "won": false,
            "over": false,
            "error": Value::Null,
            "done": false,
        });
        while self.faithful_steps - before < max_steps {
            out = self.run_mode_step()?;
            if out["done"].as_bool() == Some(true) || out["over"].as_bool() == Some(true) {
                break;
            }
        }
        Ok(json!({
            "state": out["state"],
            "steps": self.faithful_steps - before,
            "victory": out["victory"],
            "won": out["won"],
            "over": out["over"],
            "error": out["error"],
        }))
    }

    // ── autobattle ──────────────────────────────────────────────────────

    /// `autobattle_advance`: auto-play to the next run-level decision in
    /// ONE call, then hand over (the skip path).
    pub fn autobattle_advance(&mut self, max_steps: u64) -> Result<Value, SessionError> {
        if self.game.is_none() {
            return Err(SessionError::new(400, "no live run"));
        }
        let mut played = 0_u64;
        let (mut over, mut won) = (false, false);
        let mut error: Option<String> = None;
        while played < max_steps {
            let state = &self.active()?.state;
            if state.get("decision").and_then(Value::as_str) == Some("game_over") {
                over = true;
                break;
            }
            if runmode::is_stop_point(state) {
                break;
            }
            let out = self.run_mode_step()?;
            played += 1;
            self.autobattle_autoplayed += 1;
            error = out["error"].as_str().map(str::to_string);
            if error.is_some() {
                break;
            }
            if out["over"].as_bool() == Some(true) || out["done"].as_bool() == Some(true) {
                over = out["over"].as_bool().unwrap_or(false);
                won = out["victory"].as_bool().unwrap_or(false);
                break;
            }
        }
        let mut settled = self.autobattle_settle(over, won, error, true)?;
        settled["autoplayed"] = json!(played);
        Ok(settled)
    }

    /// `autobattle_step`: advance by AT MOST one decision (the watch path).
    pub fn autobattle_step(&mut self) -> Result<Value, SessionError> {
        if self.game.is_none() {
            return Err(SessionError::new(400, "no live run"));
        }
        let (mut over, mut won, mut played) = (false, false, false);
        let mut error: Option<String> = None;
        let mut step = json!({});
        let decision = self
            .active()?
            .state
            .get("decision")
            .and_then(Value::as_str)
            .map(str::to_string);
        if decision.as_deref() == Some("game_over") {
            over = true;
        } else if !runmode::is_stop_point(&self.active()?.state) {
            step = self.run_mode_step()?;
            played = true;
            self.autobattle_autoplayed += 1;
            error = step["error"].as_str().map(str::to_string);
            if step["over"].as_bool() == Some(true) || step["done"].as_bool() == Some(true) {
                over = step["over"].as_bool().unwrap_or(false);
                won = step["victory"].as_bool().unwrap_or(false);
            }
        }
        let mut settled = self.autobattle_settle(over, won, error, false)?;
        settled["played"] = json!(played);
        settled["search"] = step.get("search").cloned().unwrap_or(Value::Null);
        settled["debug"] = step.get("debug").cloned().unwrap_or(Value::Null);
        Ok(settled)
    }

    /// `_autobattle_settle`: is this a stop, and what changed?
    fn autobattle_settle(
        &mut self,
        over: bool,
        won: bool,
        error: Option<String>,
        always_diff: bool,
    ) -> Result<Value, SessionError> {
        let state = self.active()?.state.clone();
        let stopped = error.is_none() && !over && runmode::is_stop_point(&state);
        let mut diff = Value::Null;
        if always_diff || stopped || over {
            let prev = self
                .autobattle_prev_state
                .as_ref()
                .map(|p| resolve_state_names(Some(p), self.loc));
            diff = runmode::context_diff(
                prev.as_ref(),
                &resolve_state_names(Some(&state), self.loc),
                self.autobattle_autoplayed,
            );
        }
        self.autobattle_pending = if stopped {
            self.capture_champion_pick()
        } else {
            None
        };
        if stopped {
            self.autobattle_prev_state = Some(state.clone());
            self.autobattle_autoplayed = 0;
        }
        Ok(json!({
            "state": state,
            "diff": diff,
            "stopped": stopped,
            "over": over,
            "won": won,
            "error": error,
        }))
    }

    /// `_capture_champion_pick`: the off-combat brain's pick at the active
    /// node, with a display label and the option probabilities.
    fn capture_champion_pick(&mut self) -> Option<Value> {
        let node_id = self.active_id?;
        #[cfg(feature = "e2e")]
        if self.e2e_drives() {
            if let Some(ev) = self.e2e_evaluate() {
                let options: Vec<Value> = ev
                    .options
                    .iter()
                    .map(|o| {
                        json!({"action": o["action"], "args": o["args"], "label": o["label"], "prob": o["prob"]})
                    })
                    .collect();
                return Some(json!({
                    "node_id": node_id,
                    "action": ev.pick["action"],
                    "args": ev.pick["args"],
                    "label": ev.pick["label"],
                    "options": options,
                }));
            }
        }
        let rec = self.brain_recommend()?;
        let named = resolve_state_names(Some(&self.nodes[&node_id].state), self.loc);
        let action = rec["action"].as_str().unwrap_or("").to_string();
        let args = rec.get("args").cloned().unwrap_or_else(|| json!({}));
        let label = action_label(&named, &action, &args);
        Some(json!({
            "node_id": node_id,
            "action": action,
            "args": args,
            "label": label,
            "options": self.capture_option_probs(),
        }))
    }

    /// `_capture_option_probs`: per-option policy probabilities at the
    /// active node under the `ppo` brain, else `[]`.
    fn capture_option_probs(&mut self) -> Vec<Value> {
        if self.off_combat_brain != "ppo" {
            return Vec::new();
        }
        let Some(panel) = self.ppo_policy_panel() else {
            return Vec::new();
        };
        if panel["ppo_available"].as_bool() != Some(true) {
            return Vec::new();
        }
        panel["options"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|o| !o["prob"].is_null())
            .map(|o| {
                json!({
                    "action": o["action"],
                    "args": o["args"],
                    "label": o["label"],
                    "prob": o["prob"],
                })
            })
            .collect()
    }

    /// `_prob_of`: the policy probability the panel gave `action`.
    fn prob_of(options: &[Value], action: &str, args: &Value) -> Option<f64> {
        let want = native_action(Some(action), args);
        options
            .iter()
            .find(|o| {
                o["action"].as_str() == Some(action)
                    && native_action(o["action"].as_str(), &o["args"]) == want
            })
            .and_then(|o| o["prob"].as_f64())
    }

    /// `autobattle_verdict`: reveal the champion's pick for the stop the
    /// human just committed — agreement only, never a win probability.
    pub fn autobattle_verdict(&mut self) -> Result<Value, SessionError> {
        if self.game.is_none() {
            return Err(SessionError::new(400, "no live run"));
        }
        let Some(pending) = self.autobattle_pending.clone() else {
            return Ok(json!({"available": false, "reason": "no graded decision pending"}));
        };
        let node = self.active()?;
        let Some(action) = node.action.clone() else {
            return Ok(
                json!({"available": false, "reason": "no action taken at the graded decision"}),
            );
        };
        if node.parent_id != pending["node_id"].as_u64() {
            return Ok(
                json!({"available": false, "reason": "no action taken at the graded decision"}),
            );
        }
        let pending_action = pending["action"].as_str().unwrap_or("").to_string();
        let agreed = action == pending_action
            && native_action(Some(&action), &node.args)
                == native_action(Some(&pending_action), &pending["args"]);
        let options: Vec<Value> = pending["options"].as_array().cloned().unwrap_or_default();
        let champion_prob = Self::prob_of(&options, &pending_action, &pending["args"]);
        let your_prob = Self::prob_of(&options, &action, &node.args);
        let margin = match (champion_prob, your_prob) {
            (Some(c), Some(y)) => Some((c - y).max(0.0)),
            _ => None,
        };
        let your_label = node.label.clone();
        self.autobattle_pending = None;
        Ok(json!({
            "available": true,
            "agreed": agreed,
            "band": runmode::verdict_band(agreed, margin),
            "margin": margin,
            "champion_prob": champion_prob,
            "your_prob": your_prob,
            "champion_label": pending["label"],
            "your_label": your_label,
            "brain": if self.e2e_drives() { "e2e" } else { self.off_combat_brain.as_str() },
        }))
    }

    // ── "play best" brains ──────────────────────────────────────────────

    /// `_combat_choice_recommend`: the combat MCTS's pick for an open
    /// in-combat choice (`choice_ranker.best_action` at its default config).
    fn combat_choice_recommend(&self) -> Option<Value> {
        let node = self.active().ok()?;
        let game = self.game.as_ref()?;
        let rm = game.native_run()?;
        let action = best_action(rm, &json!({}))?;
        let options = normalize_options(&node.state);
        game.verb_for(action, &options)
            .map(|(verb, args)| json!({"action": verb, "args": args, "policy": "combat_mcts"}))
    }

    /// `_heuristic_recommend`: the tier-1 heuristic's `{"action", "args"}`
    /// for the active off-combat decision.
    fn heuristic_recommend(&self) -> Option<Value> {
        use sts2_core::full_run::RunPolicy;
        use sts2_runai::policy::HeuristicPolicy;
        let node = self.active().ok()?;
        let game = self.game.as_ref()?;
        let rm = game.native_run()?;
        if rmx::phase_str(rm) == "combat" || rm.is_over() {
            return None;
        }
        let actions = rmx::available_actions(rm);
        if actions.is_empty() {
            return None;
        }
        let i = HeuristicPolicy::new().choose(rm, &actions)?;
        let options = normalize_options(&node.state);
        game.verb_for(actions[i], &options)
            .map(|(verb, args)| json!({"action": verb, "args": args}))
    }

    /// `_ppo_recommend`: the greedy PPO `{"action", "args", "policy": "ppo"}`
    /// for the active off-combat decision, or `None` when PPO can't decide.
    fn ppo_recommend(&mut self) -> Option<Value> {
        let net_error = self.net_error();
        let game = self.game.as_mut()?;
        let rm = game.native_run_mut()?;
        let ev = ppo::evaluate(self.ppo.as_ref(), &net_error, rm)?;
        let chosen = ev.chosen?;
        let (verb, args) =
            runmode::resolve_workbench_verb(rm, &ev.actions[chosen], &ev.full_actions)?;
        Some(json!({"action": verb, "args": args, "policy": "ppo"}))
    }

    /// `recommended_action`: the policy's `{"action", "args"}` for the
    /// active decision — an open in-combat choice from the combat MCTS,
    /// otherwise the `ppo` brain (flagged `fallback: true` when the
    /// heuristic answered for it) or the heuristic.
    pub fn recommended_action(&mut self) -> Option<Value> {
        self.active_id?;
        // The e2e student answers every decision, combat included.
        #[cfg(feature = "e2e")]
        if self.is_e2e_brain() {
            if let Some(ev) = self.e2e_evaluate() {
                return Some(ev.pick);
            }
        }
        if let Some(rec) = self.combat_choice_recommend() {
            return Some(rec);
        }
        if self.off_combat_brain == "ppo" {
            if let Some(rec) = self.ppo_recommend() {
                return Some(rec);
            }
            let mut out = self.heuristic_recommend()?;
            out["policy"] = json!("ppo");
            out["fallback"] = json!(true);
            return Some(out);
        }
        self.heuristic_recommend()
    }

    /// `_driving_first_col`: the column of the next map room the ACTIVE
    /// run policy would take — `None` unless PPO is driving, the decision
    /// is a map choice, and the net's pick resolves to a map node. Seeds
    /// the map overlay's route so the pencil line continues the move that
    /// will happen instead of the heuristic planner's own opinion.
    pub(crate) fn driving_first_col(&mut self, state: Option<&Value>) -> Option<i32> {
        // Under the e2e advisor the student seeds the route — but only where
        // the active node IS the live position (`map`); a new node's state is
        // re-seeded once it is active (`e2e_reseed_map`).
        #[cfg(feature = "e2e")]
        if self.advisor_is_e2e() && !self.is_e2e_brain() {
            return if state.is_none() {
                self.e2e_map_col()
            } else {
                None
            };
        }
        if self.off_combat_brain != "ppo" {
            return None;
        }
        let decision = match state {
            Some(st) => st.get("decision").cloned(),
            None => self.active_id.map(|id| {
                self.nodes[&id]
                    .state
                    .get("decision")
                    .cloned()
                    .unwrap_or(Value::Null)
            }),
        };
        if decision.as_ref().and_then(Value::as_str) != Some("map_select") {
            return None;
        }
        let rec = self.ppo_recommend()?;
        if rec["action"].as_str() != Some("select_map_node") {
            return None;
        }
        rec["args"]["col"]
            .as_i64()
            .and_then(|c| i32::try_from(c).ok())
    }

    /// `ppo_policy_panel`: masked PPO softmax + V(s) for the active
    /// non-combat decision, labelled for the UI. `None` when the panel
    /// doesn't apply (combat / game over / no active node); an
    /// `ppo_available: false` payload when the net can't load.
    pub fn ppo_policy_panel(&mut self) -> Option<Value> {
        let node_id = self.active_id?;
        let net_error = self.net_error();
        let ev = {
            let game = self.game.as_mut()?;
            let rm = game.native_run_mut()?;
            ppo::evaluate(self.ppo.as_ref(), &net_error, rm)?
        };
        let parent_named = resolve_state_names(Some(&self.nodes[&node_id].state), self.loc);
        let game = self.game.as_ref()?;
        let rm = game.native_run()?;
        let deck = if ev.surface == "shop" {
            emit::run_manager::master_deck_detailed(rm)
        } else {
            Vec::new()
        };
        Some(ppo::panel_value(
            &ev,
            &self.ppo_impl,
            &deck,
            |native: &RunAction| runmode::resolve_workbench_verb(rm, native, &ev.full_actions),
            |verb, args| action_label(&parent_named, verb, args),
        ))
    }

    // ── WPA review / verify ─────────────────────────────────────────────

    /// `_wpa_rows`: scored decision rows along the active path.
    fn wpa_rows(&self) -> Vec<Value> {
        let Some(active) = self.active_id else {
            return Vec::new();
        };
        let path = self.path_to(active);
        let mut rows = Vec::new();
        let (mut floor, mut act) = (Value::Null, Value::Null);
        for (i, id) in path.iter().enumerate() {
            let node = &self.nodes[id];
            if let Some(f) = node.state.get("floor").filter(|v| !v.is_null()) {
                floor = f.clone();
            }
            if let Some(a) = node.state.get("act").filter(|v| !v.is_null()) {
                act = a.clone();
            }
            let Some(value) = node.value else {
                continue;
            };
            let taken = path
                .get(i + 1)
                .map_or(Value::Null, |next| json!(self.nodes[next].label));
            rows.push(json!({
                "node_id": node.id,
                "idx": node.idx,
                "decision": node.decision,
                "floor": floor,
                "act": act,
                "value": value,
                "taken_label": taken,
            }));
        }
        rows
    }

    /// `wpa_review`: the value curve + ranked turning points over the
    /// active line; `available: false` with a reason when fewer than two
    /// decisions carry a value.
    pub fn wpa_review(&self, config: &Value) -> Result<Value, SessionError> {
        if self.active_id.is_none() {
            return Err(SessionError::new(400, "no live run"));
        }
        let top_k = config
            .get("top_k")
            .and_then(Value::as_i64)
            .map_or(runmode::WPA_DEFAULT_TOP_K, |k| {
                usize::try_from(k).unwrap_or(0)
            });
        let min_swing = config
            .get("min_swing")
            .and_then(Value::as_f64)
            .unwrap_or(runmode::WPA_DEFAULT_MIN_SWING);
        let won = if self.active()?.state.get("decision").and_then(Value::as_str)
            == Some("game_over")
            && self.game.as_ref().and_then(|g| g.native_run()).is_some()
        {
            Some(self.native_won())
        } else {
            None
        };
        Ok(runmode::wpa_review(
            &self.wpa_rows(),
            top_k,
            min_swing,
            won,
            "no run-level value data — value recording needs the rust backend and a loadable PPO checkpoint",
        ))
    }

    /// `wpa_verify`: counterfactually check a WPA turning point — agent-
    /// finish the run from the flagged decision along the played line and
    /// along the PPO panel's preferred other option (paired seeds), compare
    /// the real outcomes. The original active leaf is restored afterwards;
    /// the two continuation branches stay in the tree.
    ///
    /// 404 unknown node; 409 a leaf with no follow-up move / no alternative
    /// (Python's `ValueError`); 400 no live run / the engine rejected the
    /// alternative (`RuntimeError`).
    pub fn wpa_verify(
        &mut self,
        node_id: u64,
        max_steps: u64,
        max_seconds: f64,
        clock: Clock<'_>,
    ) -> Result<Value, SessionError> {
        if self.game.is_none() || self.active_id.is_none() {
            return Err(SessionError::new(400, "no live run"));
        }
        if !self.nodes.contains_key(&node_id) {
            return Err(SessionError::new(404, format!("unknown node: {node_id}")));
        }
        if self.game.as_ref().and_then(|g| g.native_run()).is_none() {
            return Err(SessionError::new(400, "no live run"));
        }
        let decision_node = &self.nodes[&node_id];
        let Some(&played_child_id) = decision_node.child_ids.first() else {
            return Err(SessionError::new(
                409,
                format!("node {node_id} has no follow-up move to compare against"),
            ));
        };
        let decision = decision_node.decision.clone();
        let played_child = self.nodes[&played_child_id].clone();
        let original_active = self.active_id.expect("checked");
        let out = self.wpa_verify_arms(node_id, &played_child, max_steps, max_seconds, clock);
        // `finally`: leave the tree usable even if the restore replay fails.
        let restored = self.seek(original_active);
        let (played, alternative) = out?;
        restored?;
        let both_over =
            played["over"].as_bool() == Some(true) && alternative["over"].as_bool() == Some(true);
        let base = "one agent-played continuation per line (same-seeded combat search) — \
                    a paired spot check, not a probability";
        let (verdict, note) = if !both_over {
            (
                "same",
                format!("inconclusive — a continuation didn't finish; {base}"),
            )
        } else if played["won"] == alternative["won"] {
            ("same", base.to_string())
        } else if alternative["won"].as_bool() == Some(true) {
            ("alternative_better", base.to_string())
        } else {
            ("played_better", base.to_string())
        };
        Ok(json!({
            "node_id": node_id,
            "decision": decision,
            "played": played,
            "alternative": alternative,
            "verdict": verdict,
            "note": note,
        }))
    }

    fn fresh_agent(&mut self) {
        let seed = runmode::hash_seed(&self.seed);
        self.agent = Some(RunAgent::new(seed, self.combat_sims, self.combat_boss_sims));
    }

    /// `_finish`: agent-play the current line out, bounded by steps and,
    /// with a clock, seconds.
    fn finish_arm(
        &mut self,
        max_steps: u64,
        max_seconds: f64,
        clock: Clock<'_>,
    ) -> Result<Value, SessionError> {
        let mut steps = 0_u64;
        let deadline = clock.map(|now| now() + max_seconds * 1000.0);
        let mut out = json!({"won": false, "over": false, "error": Value::Null});
        let mut timed_out = false;
        while steps < max_steps {
            if let (Some(now), Some(deadline)) = (clock, deadline) {
                if now() > deadline {
                    timed_out = true;
                    break;
                }
            }
            out = self.run_mode_step()?;
            steps += 1;
            if out["done"].as_bool() == Some(true) || out["over"].as_bool() == Some(true) {
                break;
            }
        }
        let over = out["over"].as_bool() == Some(true);
        let mut error = out.get("error").cloned().unwrap_or(Value::Null);
        if timed_out {
            error = json!(format!(
                "verify arm timed out after {}s ({steps} steps)",
                crate::render::py::float_repr(max_seconds).trim_end_matches(".0")
            ));
        } else if !over && steps >= max_steps && error.is_null() {
            error = json!(format!(
                "verify arm hit the {max_steps}-step cap without ending"
            ));
        }
        Ok(json!({
            "won": out["won"].as_bool() == Some(true),
            "over": over,
            "steps": steps,
            "error": error,
        }))
    }

    fn wpa_verify_arms(
        &mut self,
        node_id: u64,
        played_child: &crate::session::Node,
        max_steps: u64,
        max_seconds: f64,
        clock: Clock<'_>,
    ) -> Result<(Value, Value), SessionError> {
        // Arm 1 — the played line: continue from the state AFTER the move
        // the player actually made at this decision.
        self.fresh_agent();
        self.seek(played_child.id)?;
        let mut played = self.finish_arm(max_steps, max_seconds, clock)?;
        played["label"] = json!(played_child.label);

        // Arm 2 — the alternative: back to the decision itself, the PPO
        // panel's preferred other option, finish.
        self.seek(node_id)?;
        let panel = self.ppo_policy_panel();
        let mut options: Vec<Value> = panel
            .as_ref()
            .and_then(|p| p["options"].as_array().cloned())
            .unwrap_or_default();
        options.sort_by(|a, b| {
            let ka = (a["prob"].is_null(), -a["prob"].as_f64().unwrap_or(0.0));
            let kb = (b["prob"].is_null(), -b["prob"].as_f64().unwrap_or(0.0));
            ka.0.cmp(&kb.0)
                .then(ka.1.partial_cmp(&kb.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        let played_native = native_action(played_child.action.as_deref(), &played_child.args);
        let alt = options.into_iter().find(|o| {
            native_action(o["action"].as_str(), &o["args"]) != played_native
                || o["label"] != json!(played_child.label)
        });
        let Some(alt) = alt else {
            return Err(SessionError::new(
                409,
                "no alternative option available at this decision",
            ));
        };
        self.fresh_agent();
        let action = alt["action"].as_str().unwrap_or("").to_string();
        let result = self.apply(&action, &alt["args"])?;
        if let Some(err) = result.error {
            return Err(SessionError::new(
                400,
                format!("alternative rejected by the engine: {err}"),
            ));
        }
        let mut alternative = self.finish_arm(max_steps, max_seconds, clock)?;
        alternative["label"] = alt["label"].clone();
        Ok((played, alternative))
    }
}

impl From<GameError> for SessionError {
    fn from(e: GameError) -> Self {
        SessionError::new(400, e.0)
    }
}
