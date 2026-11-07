//! The `e2e` brain on the [`Session`]: the e2e Ironclad student plays every
//! decision of the live run (combat included) — `e2e_step` is
//! `agent_step`'s twin, `e2e_recommend` feeds the auto-action and the
//! autobattle's champion pick.
//!
//! ## The student's per-run state
//!
//! The student reads more than the run: the env driver's fight tracking and
//! intent log, `grant_picked`, and the tokenizer's per-floor map read
//! ([`E2eState`]). The session keeps one state per NODE
//! (`E2eSlot::snaps`): the state at that node's position, before the env
//! looked at it. So:
//!
//! - **`e2e_step`** takes the active node's state, peeks the live run with it
//!   (`Driver::peek`), applies the forced step or the student's pick through
//!   the driver (`Driver::apply_with`, exactly as `E2eEnv` would) and stores
//!   the result under the new node. A run the student plays from the start is
//!   `E2eEnv`'s run, step for step (`tests/e2e_brain.rs`).
//! - **A move the human makes** (`apply`) or a recorded native move replayed
//!   by a restore (`apply_native`) goes through the same peek, then
//!   `Driver::note_external`. Exact except one corner: a CONFIRM that closes
//!   a multi-pick card grant counts as a pick (harmless: the next action
//!   resets the count).
//! - **seek / rewind** land on nodes that already carry their state.
//! - **A node without a state** (a cold branch after a restore, or the brain
//!   switched to `e2e` mid-run) is rebuilt by [`Session::e2e_rebuild`]: a
//!   shadow copy of the run is replayed from the deepest node that has a
//!   state, peeking at every position on the way.
//!
//! ## The student as an advisor
//!
//! A hand-played run can take the student as its advisor (`advisor: "e2e"`,
//! [`Session::set_advisor`]): [`Session::e2e_advice`] is the student's
//! evaluation of the active node — every merged class with its softmax
//! probability, its workbench verb / args and a readable label, plus the
//! combat rows in the hint's `Score` shape — cached per node, so the hint,
//! the policy panel, the choice ranker, the auto-action and the choice coach
//! of one position share ONE forward. The per-node states above are kept
//! for every human move while the student advises (`e2e_pre_external`), and
//! a run the human played from the start rebuilds them by shadow replay the
//! first time it is asked. A position the env driver cannot read (an
//! unencodable action, a cap) answers `available: false` with the reason and
//! does NOT poison the positions after it (the driver's terminal is sticky,
//! so the tracked state skips the peek that ended it).

use serde_json::{json, Value};
use sts2_e2e::contracts as ct;

use crate::e2e::{self, ApplyKind, Driver, E2eState, GrantMark, MsClock, Next};
use crate::game::Game;
use crate::localization::resolve_state_names;
use crate::render::{describe_action, NATIVE_ACTION};
use crate::rmx;
use crate::runmode;
use crate::session::{Session, SessionError};
use crate::statediff::diff_states;

/// The e2e brain's slot on a session: the net, why there is none, and the
/// student's state per node.
#[derive(Debug, Default)]
pub struct E2eSlot {
    pub(crate) net: Option<e2e::Net>,
    pub(crate) error: Option<String>,
    pub(crate) snaps: std::collections::BTreeMap<u64, E2eState>,
    /// The advisor's evaluation per node ([`Session::e2e_advice`]).
    pub(crate) advice: std::collections::BTreeMap<u64, Value>,
}

/// What [`Session::e2e_pre_external`] saw before a move it did not choose.
#[derive(Debug)]
pub(crate) struct E2ePre {
    st: E2eState,
    combat: bool,
    mark: GrantMark,
}

/// The student's pick at the active node, as workbench verbs.
#[derive(Debug, Clone)]
pub(crate) struct E2eEval {
    /// `{"action", "args", "policy": "e2e", "label", "forced"}`.
    pub pick: Value,
    /// One per merged class: `{"action", "args", "label", "prob"}`; empty
    /// for a forced step.
    pub options: Vec<Value>,
}

/// Peek `rm` with the tracked state as the env would, and keep what the
/// peek learned (fight tracking, the intent log, the decision count, the
/// tokenizer's map read) — EXCEPT a terminal on a run that is not over (an
/// unencodable position, a cap): the driver's terminal is sticky, and a
/// human goes on playing past a position the student cannot read. Returns
/// whether the move made here is a combat one.
fn peek_tracked(st: &mut E2eState, rm: &mut sts2_core::run_manager::RunManager) -> bool {
    let mut probe = st.clone();
    let next = probe.driver.peek(rm);
    match &next {
        Next::Terminal(_) if !rm.is_over() => rmx::phase_str(rm) == "combat",
        _ => {
            if let Next::Decision(d) = &next {
                sts2_e2e::tokens::prime(d, rm, &mut probe.cache);
            }
            *st = probe;
            e2e::kind_of(&next) == Some(ApplyKind::Combat)
        }
    }
}

/// A combat class's `Score.stable_idx`, the hand / potion-tray key the board
/// overlays read: a card play is its hand index, `end_turn` is `-1`, potion
/// slot `s` is `-100 - s`; any other kind (an in-combat choice) gets the
/// `-300 - class` sentinel the board does not dispatch.
fn stable_idx_of(verb: &str, args: &Value, class: usize) -> i64 {
    match verb {
        "play_card" => args["card_index"].as_i64().unwrap_or(0),
        "end_turn" => -1,
        "use_potion" => -100 - args["potion_index"].as_i64().unwrap_or(0),
        _ => -300 - i64::try_from(class).unwrap_or(0),
    }
}

/// The workbench verb for a run action, remapped onto the live legal list
/// the way the advisor sends it to `/api/live/action`.
fn verb_of(
    rm: &sts2_core::run_manager::RunManager,
    acts: &[sts2_core::run_manager::RunAction],
    a: &Value,
) -> (String, Value) {
    match rmx::action_from_value(a) {
        Ok(ra) => runmode::resolve_workbench_verb(rm, &ra, acts)
            .unwrap_or_else(|| runmode::to_workbench_verb(rm, a)),
        Err(_) => runmode::to_workbench_verb(rm, a),
    }
}

/// One class as a hint `Score` row: its probability as `win_prob` / `prob`,
/// `2p - 1` as `q` and as the visit share (`visits` = p × 1e6), keyed by
/// the hand / potion index the board overlays read (`stable_idx_of`).
fn score_row(
    rank: usize,
    class: usize,
    verb: &str,
    args: &Value,
    label: &Value,
    p: f64,
    is_best: bool,
) -> Value {
    // Visits carry the probability, so the panel's share column reads it;
    // rounding to a count is the only loss.
    #[allow(clippy::cast_possible_truncation)]
    let visits = (p * 1e6).round() as i64;
    json!({
        "stable_idx": stable_idx_of(verb, args, class),
        "action": label,
        "win_prob": p,
        // The UI reads a row with `prob` as the student's (a policy
        // probability, not a win chance).
        "prob": p,
        "q": 2.0 * p - 1.0,
        "visits": visits,
        "is_best": is_best,
        "q_saturated": false,
        "low_confidence": false,
        "target_index": args.get("target_index").cloned().unwrap_or(Value::Null),
        "pick_rank": rank,
        "in_band": false,
        "pick_reason": if is_best { json!("Instinct greedy") } else { Value::Null },
    })
}

/// `{advisor, node_id, available: false, reason, …}`: the advisor has no
/// pick at this node.
fn unavailable(node_id: u64, reason: impl Into<String>) -> Value {
    json!({
        "advisor": "e2e",
        "node_id": node_id,
        "available": false,
        "reason": reason.into(),
        "forced": false,
        "is_combat": false,
        "pick": Value::Null,
        "options": [],
        "scores": [],
    })
}

fn label_of(state_named: &Value, verb: &str, args: &Value) -> String {
    describe_action(
        Some(state_named),
        Some(&json!({"cmd": "action", "action": verb, "args": args})),
    )
}

impl Session {
    /// Whether the `e2e` brain drives this session.
    #[must_use]
    pub fn is_e2e_brain(&self) -> bool {
        self.off_combat_brain == "e2e"
    }

    /// Whether the student makes this run's AI moves (`run_mode_step`, so
    /// play to end and the autobattle's fights, and the autobattle's
    /// champion pick): it is the run's brain, or it advises a run the human
    /// plays. The AI then plays on as the advisor the human chose, not the
    /// search. A run-mode ("watch") run keeps the brain it was started with.
    #[must_use]
    pub fn e2e_drives(&self) -> bool {
        self.is_e2e_brain() || (self.advisor_is_e2e() && !self.run_mode_run)
    }

    /// Attach (or detach) the student and the reason there is none.
    pub fn set_e2e(&mut self, net: Option<e2e::Net>, error: Option<String>) {
        self.e2e.net = net;
        self.e2e.error = error;
    }

    /// Whether the student is attached.
    #[must_use]
    pub fn has_e2e(&self) -> bool {
        self.e2e.net.is_some()
    }

    fn e2e_clock(&self) -> MsClock {
        self.perf_clock.unwrap_or(e2e::fallback_ms)
    }

    /// The student, or [`e2e::NO_NET_STATUS`] (the worker fetches it and
    /// retries).
    fn e2e_net(&self) -> Result<e2e::Net, SessionError> {
        self.e2e.net.clone().ok_or_else(|| {
            SessionError::new(
                e2e::NO_NET_STATUS,
                e2e::missing_net_error(self.e2e.error.as_deref()),
            )
        })
    }

    /// Whether the student keeps its per-node state as the run goes: it
    /// drives the run, or it advises the human playing it.
    fn e2e_tracks(&self) -> bool {
        self.is_e2e_brain() || self.advisor_is_e2e()
    }

    /// 400 unless the student can read this run: Ironclad on its build.
    pub(crate) fn e2e_check_run(&self) -> Result<(), SessionError> {
        self.e2e_check_character()?;
        if self.game_version.as_str() == e2e::GAME_VERSION {
            Ok(())
        } else {
            Err(SessionError::new(
                400,
                format!(
                    "Instinct plays {} only (this run is {})",
                    e2e::GAME_VERSION,
                    self.game_version.as_str()
                ),
            ))
        }
    }

    fn e2e_check_character(&self) -> Result<(), SessionError> {
        if self.character.eq_ignore_ascii_case(e2e::CHARACTER) {
            Ok(())
        } else {
            Err(SessionError::new(
                400,
                format!(
                    "Instinct plays {} only (this run is {})",
                    e2e::CHARACTER,
                    self.character
                ),
            ))
        }
    }

    /// The integer seed the run was started with, as `E2eEnv` sees it. A
    /// run-mode session stores that integer as a decimal string (it was
    /// already hashed by the caller), so it must be parsed, not re-hashed.
    fn e2e_run_seed(&self) -> i64 {
        let seed = if self.run_mode_run {
            self.seed
                .parse::<i32>()
                .unwrap_or_else(|_| runmode::hash_seed(&self.seed))
        } else {
            runmode::hash_seed(&self.seed)
        };
        i64::from(seed)
    }

    /// Seed the root's state (a fresh run the student will play).
    pub(crate) fn e2e_seed_root(&mut self) {
        if let Some(root) = self.active_id {
            let seed = self.e2e_run_seed();
            self.e2e.snaps.insert(root, E2eState::new(seed));
        }
    }

    /// The student's state at the active node (rebuilt when missing).
    fn e2e_state_at_active(&mut self) -> Result<E2eState, SessionError> {
        let active = self.active()?.id;
        if !self.e2e.snaps.contains_key(&active) {
            self.e2e_rebuild(active)?;
        }
        Ok(self.e2e.snaps[&active].clone())
    }

    /// Build the missing states on the path to `target`: replay a shadow
    /// copy of the run from the deepest node that has one, peeking at every
    /// position the way the env would have.
    pub(crate) fn e2e_rebuild(&mut self, target: u64) -> Result<(), SessionError> {
        let path = self.path_to(target);
        let from = path.iter().rposition(|id| self.e2e.snaps.contains_key(id));
        let (from, mut st) = if let Some(i) = from {
            (i, self.e2e.snaps[&path[i]].clone())
        } else {
            let st = E2eState::new(self.e2e_run_seed());
            self.e2e.snaps.insert(path[0], st.clone());
            (0, st)
        };
        let mut shadow = self.construct_game()?;
        for &id in &path[1..=from] {
            self.replay_on(&mut shadow, id)?;
        }
        for &id in &path[from + 1..] {
            let rm = shadow
                .native_run_mut()
                .ok_or_else(|| SessionError::new(409, "no native run"))?;
            let mark = Driver::grant_mark(rm);
            let combat = peek_tracked(&mut st, rm);
            self.replay_on(&mut shadow, id)?;
            let rm = shadow
                .native_run_mut()
                .ok_or_else(|| SessionError::new(409, "no native run"))?;
            st.driver.note_external(rm, mark, combat);
            self.e2e.snaps.insert(id, st.clone());
        }
        Ok(())
    }

    /// Re-apply node `id`'s action on a shadow game (no display logs).
    fn replay_on(&self, game: &mut Game, id: u64) -> Result<(), SessionError> {
        let n = &self.nodes[&id];
        let action = n.action.clone().unwrap_or_default();
        if action == NATIVE_ACTION {
            let rm = game
                .native_run_mut()
                .ok_or_else(|| SessionError::new(409, "no native run"))?;
            let res = rmx::take_native_value(rm, &n.args, false, false).map_err(|e| {
                SessionError::new(409, format!("e2e rebuild diverged at node {id}: {e}"))
            })?;
            if res != "advanced" && res != "run_over" {
                return Err(SessionError::new(
                    409,
                    format!("e2e rebuild diverged at node {id}: take_action returned '{res}'"),
                ));
            }
            Ok(())
        } else {
            game.act(&action, &n.args).map(|_| ()).map_err(|e| {
                SessionError::new(409, format!("e2e rebuild diverged at node {id}: {}", e.0))
            })
        }
    }

    /// Before a move the student did not choose (the human's `apply`, a
    /// restore's `apply_native`): the active node's state, peeked at the
    /// live position. `None` unless the e2e brain drives and the node has a
    /// state (a node without one is rebuilt when the student next plays).
    pub(crate) fn e2e_pre_external(&mut self) -> Option<E2ePre> {
        if !self.e2e_tracks() {
            return None;
        }
        let mut st = self.e2e.snaps.get(&self.active_id?)?.clone();
        let rm = self.game.as_mut()?.native_run_mut()?;
        let mark = Driver::grant_mark(rm);
        let combat = peek_tracked(&mut st, rm);
        Some(E2ePre { st, combat, mark })
    }

    /// After that move landed as `node`: count it and store the state.
    pub(crate) fn e2e_post_external(&mut self, pre: Option<E2ePre>, node: u64) {
        let Some(mut pre) = pre else {
            return;
        };
        let Some(rm) = self.game.as_ref().and_then(Game::native_run) else {
            return;
        };
        pre.st.driver.note_external(rm, pre.mark, pre.combat);
        self.e2e.snaps.insert(node, pre.st);
    }

    /// The student's pick at the active node (a forced step's action, or the
    /// greedy class's representative), as workbench verbs, with the class
    /// probabilities. `None` when the run is over, the student has no net,
    /// or the pick has no workbench verb.
    pub(crate) fn e2e_evaluate(&mut self) -> Option<E2eEval> {
        let adv = self.e2e_advice().ok()?;
        if adv["available"] != json!(true) || adv["pick"].is_null() {
            return None;
        }
        let pick = adv["pick"].clone();
        let options = if pick["forced"] == json!(true) {
            Vec::new()
        } else {
            adv["options"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|o| json!({"action": o["action"], "args": o["args"], "label": o["label"], "prob": o["prob"]}))
                .collect()
        };
        Some(E2eEval { pick, options })
    }

    /// `POST /api/live/e2e-advice`: the student's evaluation of the active
    /// node, cached per node —
    ///
    /// `{advisor: "e2e", node_id, available, reason, forced, is_combat,
    /// decision_type, n_classes, ms, pick, options, scores}`. `pick` and each
    /// of `options` (one per merged class, by probability) carry the class's
    /// workbench `action` / `args` (what to send to `/api/live/action`), a
    /// readable `label`, `prob` (the softmax over the classes — a policy
    /// probability, never a win chance), `class`, `members` (the legal
    /// actions the class merges), `is_best` and the env's `run_action`.
    /// `scores` are the combat classes as the hint's `Score` rows (a card's
    /// `stable_idx` is its hand index), so the board can star them. A forced
    /// step is one option at probability 1.
    ///
    /// `available: false` (with `reason`) when the run is over, the env
    /// driver cannot read the position, or the run is not one the student
    /// plays.
    ///
    /// # Errors
    ///
    /// 424 when the net is not loaded (the host fetches it and retries); 400
    /// with no live run.
    pub fn e2e_advice(&mut self) -> Result<Value, SessionError> {
        let node_id = self.active()?.id;
        if let Some(v) = self.e2e.advice.get(&node_id) {
            return Ok(v.clone());
        }
        let net = self.e2e_net()?;
        let out = match self.e2e_check_run() {
            Err(e) => unavailable(node_id, e.message),
            Ok(()) => match self.e2e_state_at_active() {
                Err(e) => unavailable(
                    node_id,
                    format!("Instinct could not follow this run: {}", e.message),
                ),
                Ok(st) => self.e2e_advice_at(node_id, &net, st)?,
            },
        };
        self.e2e.advice.insert(node_id, out.clone());
        Ok(out)
    }

    fn e2e_advice_at(
        &mut self,
        node_id: u64,
        net: &e2e::Net,
        mut st: E2eState,
    ) -> Result<Value, SessionError> {
        let clock = self.e2e_clock();
        let named = resolve_state_names(Some(&self.nodes[&node_id].state), self.loc);
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| SessionError::new(400, "no live run"))?;
        let rm = game
            .native_run_mut()
            .ok_or_else(|| SessionError::new(400, "no live run"))?;
        let next = st.driver.peek(rm);
        let rm: &sts2_core::run_manager::RunManager = rm;
        let acts = rm.get_available_actions();
        let is_combat = rmx::phase_str(rm) == "combat";
        // (class, run action, prob, members) per class; the forced step is
        // one class at probability 1.
        let (rows, forced, dtype, ms): (Vec<(usize, Value, f64, usize)>, bool, Value, f64) =
            match next {
                Next::Terminal(t) => {
                    let reason = match (t.outcome, t.error, t.cap) {
                        ("error", Some(e), _) => {
                            format!("Instinct cannot read this position: {e}")
                        }
                        ("capped", _, Some(cap)) => {
                            format!("Instinct stops here (its {cap} cap)")
                        }
                        (outcome, _, _) => format!("the run is over ({outcome})"),
                    };
                    return Ok(unavailable(node_id, reason));
                }
                Next::Forced(f) => (vec![(0, f.action, 1.0, 1)], true, Value::Null, 0.0),
                Next::Decision(d) => {
                    let c = match e2e::decide(net, &d, rm, &mut st.cache, clock) {
                        Ok(c) => c,
                        Err(e) => {
                            return Ok(unavailable(
                                node_id,
                                format!("Instinct cannot read this position: {e}"),
                            ))
                        }
                    };
                    let rows = c
                        .probs
                        .iter()
                        .enumerate()
                        .filter_map(|(i, p)| {
                            let (a, _) = d.class_action(i).ok()?;
                            Some((i, a, *p, d.members.get(i).map_or(1, Vec::len)))
                        })
                        .collect();
                    (
                        rows,
                        false,
                        json!(ct::dtype_name(d.dtype)),
                        c.ms_tokenize + c.ms_net,
                    )
                }
            };
        let best = rows
            .iter()
            .enumerate()
            .fold(None::<(usize, f64)>, |b, (i, r)| match b {
                Some((_, p)) if p >= r.2 => b,
                _ => Some((i, r.2)),
            })
            .map(|(i, _)| rows[i].0);
        let mut options: Vec<Value> = rows
            .iter()
            .map(|(class, a, p, members)| {
                let (verb, args) = verb_of(rm, &acts, a);
                let mut label = label_of(&named, &verb, &args);
                if *members > 1 {
                    label = format!("{label} (×{members})");
                }
                json!({
                    "class": class,
                    "action": verb,
                    "args": args,
                    "label": label,
                    "prob": p,
                    "members": members,
                    "is_best": Some(*class) == best,
                    "run_action": a,
                })
            })
            .collect();
        options.sort_by(|a, b| {
            b["prob"]
                .as_f64()
                .partial_cmp(&a["prob"].as_f64())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let pick = options
            .iter()
            .find(|o| o["is_best"] == json!(true))
            .map_or(Value::Null, |o| {
                json!({
                    "action": o["action"],
                    "args": o["args"],
                    "policy": "e2e",
                    "label": o["label"],
                    "forced": forced,
                    "prob": o["prob"],
                })
            });
        let scores: Vec<Value> = if is_combat {
            options
                .iter()
                .enumerate()
                .map(|(rank, o)| {
                    let class = o["class"]
                        .as_u64()
                        .and_then(|c| usize::try_from(c).ok())
                        .unwrap_or(0);
                    score_row(
                        rank,
                        class,
                        o["action"].as_str().unwrap_or(""),
                        &o["args"],
                        &o["label"],
                        o["prob"].as_f64().unwrap_or(0.0),
                        o["is_best"] == json!(true),
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(json!({
            "advisor": "e2e",
            "node_id": node_id,
            "available": true,
            "reason": Value::Null,
            "forced": forced,
            "is_combat": is_combat,
            "decision_type": dtype,
            "n_classes": options.len(),
            "ms": ms,
            "pick": pick,
            "options": options,
            "scores": scores,
        }))
    }

    // ── the advisor's surfaces (the shapes the search advisors answer) ──

    /// `POST /api/live/hint` under the e2e advisor: the student's combat
    /// classes as `Score` rows, `{scores, pv: null, advisor: "e2e", error?}`.
    /// `None` off a `combat_play` node (the route's 409).
    ///
    /// # Errors
    ///
    /// 424 without the net.
    pub fn e2e_hint(&mut self) -> Result<Option<Value>, SessionError> {
        if self.active()?.decision != "combat_play" {
            return Ok(None);
        }
        let adv = self.e2e_advice()?;
        let mut out = json!({
            "scores": adv["scores"],
            "pv": Value::Null,
            "advisor": "e2e",
            "config": {},
        });
        if adv["available"] != json!(true) {
            out["error"] = adv["reason"].clone();
        }
        Ok(Some(out))
    }

    /// `POST /api/live/auto-action` under the e2e advisor: the student's pick
    /// (`{action, args, policy: "e2e", label, forced, prob}`), `None` when it
    /// has none here.
    ///
    /// # Errors
    ///
    /// 424 without the net.
    pub fn e2e_recommended(&mut self) -> Result<Option<Value>, SessionError> {
        let adv = self.e2e_advice()?;
        Ok(
            (adv["available"] == json!(true) && !adv["pick"].is_null())
                .then(|| adv["pick"].clone()),
        )
    }

    /// `POST /api/live/ppo-policy` under the e2e advisor: the policy panel's
    /// payload with the student's classes (`policy: "e2e"`, `value: null` —
    /// the student has no critic head here). `None` at a `combat_play` node
    /// (the hint owns those) and at game over.
    ///
    /// # Errors
    ///
    /// 424 without the net.
    pub fn e2e_policy_panel(&mut self) -> Result<Option<Value>, SessionError> {
        let decision = self.active()?.decision.clone();
        if decision == "combat_play" || decision == "game_over" {
            return Ok(None);
        }
        let adv = self.e2e_advice()?;
        let options: Vec<Value> = adv["options"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|o| {
                json!({
                    "action": o["action"],
                    "args": o["args"],
                    "label": o["label"],
                    "prob": o["prob"],
                    "is_best": o["is_best"],
                })
            })
            .collect();
        Ok(Some(json!({
            "ppo_available": adv["available"],
            "policy": "e2e",
            "impl": "e2e",
            "surface": adv.get("decision_type").cloned().unwrap_or(Value::Null),
            "value": Value::Null,
            "options": options,
            "forced": adv["forced"],
            "error": adv["reason"],
        })))
    }

    /// `POST /api/live/choice-ranker` under the e2e advisor: the ranker's
    /// rows (`{key, label, win_prob, q, visits, is_best}` per option of the
    /// in-combat card-select) with the student's class probability as
    /// `win_prob` / `prob` and `advisor: "e2e"`. `None` off a `card_select`.
    ///
    /// # Errors
    ///
    /// 424 without the net.
    pub fn e2e_rank_choice(&mut self) -> Result<Option<Value>, SessionError> {
        if self.active()?.decision != "card_select" {
            return Ok(None);
        }
        let adv = self.e2e_advice()?;
        let Some(rm) = self.game.as_ref().and_then(Game::native_run) else {
            return Err(SessionError::new(400, "no live run"));
        };
        let options: Vec<(sts2_core::run_manager::RunAction, &Value)> = adv["options"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|o| Some((rmx::action_from_value(&o["run_action"]).ok()?, o)))
            .collect();
        let rows: Vec<Value> = crate::advisors::candidates(rm)
            .into_iter()
            .map(|c| {
                let o = options
                    .iter()
                    .find(|(a, _)| *a == c.action)
                    .map(|(_, o)| *o);
                let p = o.and_then(|o| o["prob"].as_f64());
                json!({
                    "key": c.key,
                    "label": o.map_or_else(|| json!(c.label), |o| o["label"].clone()),
                    "win_prob": p,
                    "prob": p,
                    "q": p.map(|p| 2.0 * p - 1.0),
                    "visits": 0,
                    "is_best": o.is_some_and(|o| o["is_best"] == json!(true)),
                })
            })
            .collect();
        let mut out = json!({
            "candidates": rows,
            "decision": "card_select",
            "act": rm.run_state.act_index,
            "config": {},
            "advisor": "e2e",
        });
        if adv["available"] != json!(true) {
            out["error"] = adv["reason"].clone();
        }
        Ok(Some(out))
    }

    /// The column of the map node the student picks at the active
    /// `map_select` node (seeds the map overlay's route), `None` elsewhere.
    pub(crate) fn e2e_map_col(&mut self) -> Option<i32> {
        let adv = self.e2e_advice().ok()?;
        if adv["pick"]["action"].as_str() != Some("select_map_node") {
            return None;
        }
        adv["pick"]["args"]["col"]
            .as_i64()
            .and_then(|c| i32::try_from(c).ok())
    }

    /// Under the e2e advisor at a `map_select` node, re-seed `state`'s
    /// `full_map` route on the student's pick (the planner draws from it),
    /// or build the plain map when it has none. Call only once the active
    /// node is the live position.
    pub(crate) fn e2e_reseed_map(&mut self, state: &mut Value) {
        if !self.advisor_is_e2e()
            || state.get("decision").and_then(Value::as_str) != Some("map_select")
        {
            return;
        }
        let col = self.e2e_map_col();
        if col.is_none() && state.get("full_map").is_some() {
            return;
        }
        let Some(game) = self.game.as_ref() else {
            return;
        };
        // No student column: the plain map (`map_state_live` left it out).
        let full_map = game.get_map(col);
        if full_map.as_object().is_some_and(|m| !m.is_empty()) {
            state["full_map"] = full_map;
            if let Some(id) = self.active_id {
                if let Some(n) = self.nodes.get_mut(&id) {
                    n.state["full_map"] = state["full_map"].clone();
                }
            }
        }
    }

    /// The `e2e` brain's `run_mode_step`: apply ONE step of the student (a
    /// forced step or its pick) natively and record a node. 424 when the
    /// net is not loaded (nothing applied); 400 for a run the student does
    /// not play (not Ironclad, or not its build).
    pub(crate) fn e2e_step(&mut self) -> Result<Value, SessionError> {
        let net = self.e2e_net()?;
        self.e2e_check_run()?;
        let parent = self.active()?;
        let parent_id = parent.id;
        let parent_named = resolve_state_names(Some(&parent.state), self.loc);
        let mut st = self.e2e_state_at_active()?;
        let clock = self.e2e_clock();
        let marks = self.mark_display_logs();
        let next = {
            let rm = self
                .game
                .as_mut()
                .and_then(Game::native_run_mut)
                .ok_or_else(|| SessionError::new(400, "no live run"))?;
            st.driver.peek(rm)
        };
        let rm = self
            .game
            .as_ref()
            .and_then(Game::native_run)
            .ok_or_else(|| SessionError::new(400, "no live run"))?;
        let phase = rmx::phase_str(rm);
        let (action, kind, choice, decision) = match next {
            Next::Terminal(t) => {
                let error = match t.outcome {
                    "error" => t.error.clone(),
                    "capped" => Some(format!("Instinct run capped ({})", t.cap.unwrap_or("?"))),
                    _ => None,
                };
                let state = self.state()?;
                let over = state.get("decision").and_then(Value::as_str) == Some("game_over")
                    || rm_over(self);
                let won = over && rm_won(self);
                return Ok(json!({
                    "state": state,
                    "steps": self.faithful_steps,
                    "victory": won,
                    "won": won,
                    "over": over,
                    "error": error,
                    "done": true,
                    "debug": {"brain": "e2e", "summary": format!("Instinct: run {}", t.outcome)},
                    "search": Value::Null,
                }));
            }
            Next::Forced(f) => (f.action, f.kind, None, None),
            Next::Decision(d) => match e2e::decide(&net, &d, rm, &mut st.cache, clock)
                .and_then(|c| d.class_action(c.class).map(|(a, k)| (a, k, c)))
            {
                Ok((a, k, c)) => (a, k, Some(c), Some(d)),
                Err(e) => {
                    return self.e2e_step_failed(&format!("e2e: {e}"), &format!("e2e failed: {e}"))
                }
            },
        };
        // Labels BEFORE the move (the map row / bundle sub-state are pre-move).
        let (verb, wb_args) = runmode::to_workbench_verb(rm, &action);
        let act_label = label_of(&parent_named, &verb, &wb_args);
        let rows: Vec<Value> = match (&choice, &decision) {
            (Some(c), Some(d)) => class_rows(rm, d, c, &parent_named),
            _ => Vec::new(),
        };
        let result = {
            let game = self.require_game()?;
            let (rcp, rcl) = game.play_log_flags();
            let rm = game
                .native_run_mut()
                .ok_or_else(|| SessionError::new(400, "no live run"))?;
            st.driver.apply_with(rm, &action, kind, |rm, a| {
                rmx::take_native_value(rm, a, rcp, rcl)
            })
        };
        let result = match result {
            // Every replay path (`replay_node`, a rebuild, `replay_on`)
            // refuses any other result, so a node recorded for one could
            // never be reloaded; `apply_native` refuses it the same way.
            Ok(r @ ("advanced" | "run_over")) => r,
            Ok(r) => {
                let e = format!("take_action returned '{r}'");
                return self.e2e_step_failed(
                    &format!("e2e: {e}"),
                    &format!("e2e: {act_label} refused: {e}"),
                );
            }
            Err(e) => {
                return self.e2e_step_failed(
                    &format!("e2e: {e}"),
                    &format!("e2e: {act_label} refused: {e}"),
                );
            }
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
        let state = self.map_state(raw);
        let node = self.add_node(
            Some(parent_id),
            state.clone(),
            Some(NATIVE_ACTION.to_string()),
            action,
            act_label.clone(),
            details,
            combat_rows,
        );
        self.active_id = Some(node);
        self.faithful_steps += 1;
        self.e2e.snaps.insert(node, st);
        let over = rm_over(self);
        let won = over && rm_won(self);
        let is_combat = kind == ApplyKind::Combat;
        let mut debug = json!({
            "brain": "e2e",
            "phase": phase,
            "is_combat": is_combat,
            "action": act_label,
            "result": result,
            "forced": choice.is_none(),
        });
        let search = if let (Some(c), Some(d)) = (&choice, &decision) {
            let ms = c.ms_tokenize + c.ms_net;
            debug["decision_type"] = json!(ct::dtype_name(d.dtype));
            debug["n_classes"] = json!(d.n_classes());
            debug["prob"] = json!(c.probs[c.class]);
            debug["ms_tokenize"] = json!(c.ms_tokenize);
            debug["ms_net"] = json!(c.ms_net);
            debug["ms"] = json!(ms);
            debug["summary"] = json!(format!(
                "e2e {} → {act_label} (p {:.2}, {ms:.1} ms)",
                ct::dtype_name(d.dtype),
                c.probs[c.class]
            ));
            Some(json!({
                "scores": rows,
                "action": act_label,
                "enemies": parent_named.get("enemies").cloned().unwrap_or_else(|| json!([])),
            }))
        } else {
            debug["summary"] = json!(format!("e2e forced → {act_label}"));
            None
        };
        Ok(json!({
            "state": state,
            "steps": self.faithful_steps,
            "victory": won,
            "won": won,
            "over": over,
            "error": Value::Null,
            "done": false,
            "debug": debug,
            "search": search,
        }))
    }
}

impl Session {
    /// `e2e_step`'s reply when the student cannot move: nothing recorded,
    /// the run stops (`done`).
    fn e2e_step_failed(&mut self, error: &str, summary: &str) -> Result<Value, SessionError> {
        Ok(json!({
            "state": self.state()?,
            "steps": self.faithful_steps,
            "victory": false,
            "won": false,
            "over": false,
            "error": error,
            "done": true,
            "debug": {"brain": "e2e", "summary": summary},
            "search": Value::Null,
        }))
    }
}

fn rm_over(s: &Session) -> bool {
    s.game
        .as_ref()
        .and_then(Game::native_run)
        .is_some_and(sts2_core::run_manager::RunManager::is_over)
}

fn rm_won(s: &Session) -> bool {
    s.game
        .as_ref()
        .and_then(Game::native_run)
        .is_some_and(|rm| rm.run_state.player_won)
}

/// The sidebar rows: one hint `Score` row per merged class (`score_row`,
/// the same rows `e2e_advice` gives), the student's pick starred; sorted by
/// probability.
fn class_rows(
    rm: &sts2_core::run_manager::RunManager,
    d: &e2e::Decision,
    c: &e2e::Choice,
    named: &Value,
) -> Vec<Value> {
    let acts = rm.get_available_actions();
    let mut order: Vec<usize> = (0..c.probs.len()).collect();
    order.sort_by(|&a, &b| {
        c.probs[b]
            .partial_cmp(&c.probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
        .iter()
        .filter_map(|&i| d.class_action(i).ok().map(|(a, _)| (i, a)))
        .enumerate()
        .map(|(rank, (i, a))| {
            let (verb, args) = verb_of(rm, &acts, &a);
            let members = d.members.get(i).map_or(1, Vec::len);
            let mut label = label_of(named, &verb, &args);
            if members > 1 {
                label = format!("{label} (×{members})");
            }
            score_row(
                rank,
                i,
                &verb,
                &args,
                &json!(label),
                c.probs[i],
                i == c.class,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! An e2e run-mode session reproduces `E2eEnv` action for action, through
    //! seek, a rebuild of forgotten states, and a persist round trip.

    use super::*;
    use sts2_core::GameVersion;
    use sts2_e2e::env::{new_run_at, take_run_action};

    const SEED: i32 = 170_000_000;

    fn net() -> Option<e2e::Net> {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../sts2-pycli/data/e2e")
            .join(e2e::E2E_FILE);
        let bytes = std::fs::read(&p).ok()?;
        Some(e2e::load(&bytes).expect("shipped e2e weights load"))
    }

    /// Every action the env applies (forced steps included), first `n`.
    fn reference(net: &e2e::Net, n: usize) -> Vec<Value> {
        let mut rm = new_run_at(i64::from(SEED), 10);
        let mut st = E2eState::new(i64::from(SEED));
        let mut out = Vec::new();
        while out.len() < n {
            let (action, kind) = match st.driver.peek(&mut rm) {
                Next::Terminal(_) => break,
                Next::Forced(f) => (f.action, f.kind),
                Next::Decision(d) => {
                    let c =
                        e2e::decide(net, &d, &rm, &mut st.cache, e2e::fallback_ms).expect("decide");
                    d.class_action(c.class).expect("class")
                }
            };
            st.driver
                .apply_with(&mut rm, &action, kind, take_run_action)
                .expect("apply");
            out.push(action);
        }
        out
    }

    fn session(net: &e2e::Net) -> Session {
        let mut s = Session::new_for(GameVersion::V0_111_0);
        s.set_e2e_setup(true);
        s.set_e2e(Some(net.clone()), None);
        s.start_run_mode("Ironclad", SEED, 10, "e2e", "rust", 1000, 4000, 1, None)
            .expect("start");
        s
    }

    /// Step `n` times, checking each recorded native action against
    /// `want[from..]`; returns how many steps were combat decisions with rows.
    fn play(s: &mut Session, want: &[Value], from: usize, n: usize) -> usize {
        let mut rows = 0;
        for (i, w) in want.iter().enumerate().skip(from).take(n) {
            let out = s.run_mode_step().expect("step");
            assert!(out["error"].is_null(), "step {i}: {}", out["error"]);
            let node = s.active().expect("active");
            assert_eq!(node.action.as_deref(), Some(NATIVE_ACTION));
            assert_eq!(node.args, *w, "step {i}");
            assert_eq!(out["debug"]["brain"], json!("e2e"));
            if out["debug"]["is_combat"] == json!(true) && out["search"].is_object() {
                rows += 1;
                assert!(out["debug"]["ms"].as_f64().is_some());
                let scores = out["search"]["scores"].as_array().expect("scores");
                assert!(scores.len() >= 2);
                assert_eq!(
                    scores
                        .iter()
                        .filter(|r| r["is_best"] == json!(true))
                        .count(),
                    1
                );
            }
        }
        rows
    }

    #[test]
    fn e2e_session_reproduces_the_env_through_seek_rebuild_and_restore() {
        let Some(net) = net() else {
            eprintln!("skipped: no shipped e2e weights");
            return;
        };
        let want = reference(&net, 160);
        assert_eq!(want.len(), 160);
        let mut s = session(&net);
        let combat_rows = play(&mut s, &want, 0, 100);
        assert!(combat_rows > 5, "the first 100 steps include a fight");

        // Rewind into the middle and play forward again.
        let path = s.path_to(s.active().expect("active").id);
        s.seek(path[60]).expect("seek");
        play(&mut s, &want, 60, 40);

        // Forget every state but the root's: the next step rebuilds them by
        // replaying a shadow run.
        let root = path[0];
        s.e2e.snaps.retain(|id, _| *id == root);
        play(&mut s, &want, 100, 20);

        // A reload: persist, import into a fresh session, keep playing.
        let payload = s.export_persist();
        assert_eq!(payload["e2e_setup"], json!(true));
        let mut r = Session::new_for(GameVersion::V0_111_0);
        r.set_e2e(Some(net.clone()), None);
        r.import_persist(&payload, |_, _| {}).expect("import");
        assert!(r.e2e_setup());
        assert!(r.is_e2e_brain());
        play(&mut r, &want, 120, 40);
    }

    #[test]
    fn without_a_net_the_brain_asks_for_one_and_applies_nothing() {
        let mut s = Session::new_for(GameVersion::V0_111_0);
        s.set_e2e_setup(true);
        s.start_run_mode("Ironclad", SEED, 10, "e2e", "rust", 1000, 4000, 1, None)
            .expect("start");
        let before = s.active().expect("active").id;
        let err = s.run_mode_step().expect_err("no net");
        assert_eq!(err.status, e2e::NO_NET_STATUS);
        assert_eq!(s.active().expect("active").id, before);
    }

    #[test]
    fn the_brain_refuses_other_characters() {
        let Some(net) = net() else {
            return;
        };
        let mut s = Session::new_for(GameVersion::V0_111_0);
        s.set_e2e(Some(net), None);
        s.start_run_mode("Silent", 3, 0, "e2e", "rust", 1000, 4000, 1, None)
            .expect("start");
        assert_eq!(s.run_mode_step().expect_err("silent").status, 400);
    }

    #[test]
    fn the_brain_refuses_other_builds() {
        let Some(net) = net() else {
            return;
        };
        let mut s = Session::new_for(GameVersion::V0_107_1);
        s.set_e2e(Some(net), None);
        s.start_run_mode("Ironclad", 3, 0, "e2e", "rust", 1000, 4000, 1, None)
            .expect("start");
        let before = s.active().expect("active").id;
        assert_eq!(s.run_mode_step().expect_err("v0.107.1").status, 400);
        assert_eq!(s.active().expect("active").id, before);
    }

    #[test]
    fn the_student_is_seeded_with_the_runs_integer_seed() {
        let mut s = Session::new_for(GameVersion::V0_111_0);
        s.set_e2e_setup(true);
        s.start_run_mode("Ironclad", SEED, 10, "e2e", "rust", 1000, 4000, 1, None)
            .expect("start");
        assert_eq!(s.e2e_run_seed(), i64::from(SEED));
    }
}
