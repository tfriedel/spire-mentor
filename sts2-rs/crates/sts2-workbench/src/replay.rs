//! The replay viewer (M4): the routes `server.py` serves over a loaded run
//! log (`TimelineStore`), here over a [`ReplayStore`].
//!
//! Pure projections — `timeline`, `map`, `step`, the deck routes, the WPA
//! review — are the Python code line for line. The engine-backed ones —
//! `recommend`, `search_tree`, a regret review's re-search, and "play from
//! here" — rebuild the run from its recorded line: the log's head carries
//! the run identity and every action its replayable verb, so replaying the
//! line to step `idx` reproduces the exact engine state. The Python server
//! instead rebuilds a fight from the rendered state through a lossy adapter
//! (it cannot replay — its `sts2_hint` lane predates the replay line), so
//! those three are structural twins in the golden parity, as the live
//! regret coach already is.

use serde_json::{json, Value};
use sts2_core::CombatState;

use crate::advisors::{self, DecisionStep};
use crate::api::ApiError;
use crate::combatlog::format_combat_log;
use crate::deck;
use crate::enrich::{asset_url, enrich, enrich_map_nodes, log_names, EnrichCtx};
use crate::game::{Game, GameError};
use crate::localization::{resolve_state_names, Loc};
use crate::render::{describe_action, NATIVE_ACTION};
use crate::runlog::{resume_capability, ReplayStore, Step};
use crate::runmode;
use crate::statediff::diff_states;

fn deck_of(state: &Value) -> Vec<Value> {
    state
        .get("player")
        .and_then(|p| p.get("deck"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// `GET /api/timeline`.
#[must_use]
pub fn timeline(store: &ReplayStore, loc: &'static Loc) -> Value {
    let steps = store.all();
    let resolved: Vec<Value> = steps
        .iter()
        .map(|s| resolve_state_names(Some(&s.state), loc))
        .collect();
    let names = log_names(loc);
    let player_id = store.character().to_uppercase().replace(' ', "_");
    let step_rows = |i: usize| -> Vec<Value> {
        if !steps[i].combat_log.is_empty() {
            return steps[i].combat_log.clone();
        }
        let Some(next) = steps.get(i + 1) else {
            return Vec::new();
        };
        let raw0 = steps[i]
            .state
            .get("combat_log_raw")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let raw1 = next
            .state
            .get("combat_log_raw")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let tail: Vec<Value> = if raw1.len() >= raw0.len() {
            raw1[raw0.len()..].to_vec()
        } else {
            raw1
        };
        if tail.is_empty() {
            Vec::new()
        } else {
            format_combat_log(&tail, Some(&names), false, &[], &player_id)
        }
    };
    let out: Vec<Value> = steps
        .iter()
        .enumerate()
        .map(|(i, s)| {
            json!({
                "idx": s.idx,
                "decision": s.decision,
                "label": describe_action(Some(&resolved[i]), Some(&s.chosen_action)),
                "details": if i + 1 < steps.len() { diff_states(&resolved[i], &resolved[i + 1]) } else { Vec::new() },
                "combat_log": step_rows(i),
                "turn": s.meta["turn"],
                "floor": s.meta["floor"],
                "act": s.meta["act"],
                "act_name": s.meta["act_name"],
            })
        })
        .collect();
    json!({"steps": out})
}

/// `GET /api/map`: one row per `map_select` step, the taken node marked,
/// plus the full graphs the recorder captured.
#[must_use]
pub fn map(store: &ReplayStore, ctx: &EnrichCtx<'_>) -> Value {
    let mut rows = Vec::new();
    let mut graphs = Vec::new();
    for s in store.all() {
        if s.decision != "map_select" {
            continue;
        }
        let state = &s.state;
        let t_args = s.chosen_action.get("args").cloned().unwrap_or(Value::Null);
        let (tcol, trow) = (
            t_args.get("col").cloned().unwrap_or(Value::Null),
            t_args.get("row").cloned().unwrap_or(Value::Null),
        );
        let nodes: Vec<Value> = state
            .get("choices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|c| {
                let col = c.get("col").cloned().unwrap_or(Value::Null);
                let row = c.get("row").cloned().unwrap_or(Value::Null);
                let ntype = c.get("type").cloned().unwrap_or(Value::Null);
                json!({
                    "col": col,
                    "row": row,
                    "type": ntype,
                    "chosen": col == tcol && row == trow,
                    "icon": asset_url(ctx.assets_base, ctx.assets.room_icon(ntype.as_str(), None)),
                })
            })
            .collect();
        rows.push(json!({
            "idx": s.idx,
            "floor": s.meta["floor"],
            "act": s.meta["act"],
            "act_name": s.meta["act_name"],
            "nodes": nodes,
        }));
        if let Some(fm) = state
            .get("full_map")
            .filter(|f| crate::render::py::truthy(f))
        {
            graphs.push(json!({
                "idx": s.idx,
                "act": s.meta["act"],
                "act_name": s.meta["act_name"],
                "floor": s.meta["floor"],
                "full_map": enrich_map_nodes(fm.clone(), ctx.assets, ctx.assets_base),
            }));
        }
    }
    json!({"rows": rows, "graphs": graphs})
}

fn step_at(store: &ReplayStore, idx: i64) -> Result<&Step, ApiError> {
    store.get(idx).map_err(|e| ApiError::new(404, e))
}

/// `GET /api/step/{idx}`: the enriched step.
pub fn step(store: &ReplayStore, idx: i64, ctx: &EnrichCtx<'_>) -> Result<Value, ApiError> {
    Ok(enrich(step_at(store, idx)?.to_value(), ctx))
}

/// `GET /api/step/{idx}/deck-analysis`, graded with the recording's own
/// build's tables (GH #877).
pub fn step_deck_analysis(store: &ReplayStore, idx: i64) -> Result<Value, ApiError> {
    let s = step_at(store, idx)?;
    Ok(deck::analyze_deck_for(
        store.game_version(),
        &deck_of(&s.state),
        deck::DEFAULT_ENERGY,
        deck::DEFAULT_HAND,
        deck::DEFAULT_SAMPLES,
        0,
    ))
}

/// `GET /api/deck/trend`: one point per deck change along the run.
#[must_use]
pub fn deck_trend(store: &ReplayStore) -> Value {
    let version = store.game_version();
    let mut points = Vec::new();
    let mut last_sig: Option<Vec<(String, bool)>> = None;
    for s in store.all() {
        let deck = deck_of(&s.state);
        if deck.is_empty() {
            continue;
        }
        let sig = deck::deck_signature(&deck);
        if last_sig.as_ref() == Some(&sig) {
            continue;
        }
        last_sig = Some(sig);
        let report = deck::analyze_deck_for(
            version,
            &deck,
            deck::DEFAULT_ENERGY,
            deck::DEFAULT_HAND,
            60,
            0,
        );
        points.push(json!({
            "idx": s.idx,
            "floor": s.meta["floor"],
            "act": s.meta["act"],
            "deck_size": report.get("deck_size").cloned().unwrap_or(Value::Null),
            "scores": report.get("scores").cloned().unwrap_or(Value::Null),
        }));
    }
    json!({"points": points})
}

/// `POST /api/deck/reward-preview {deck, candidates, game_version?}`
/// (stateless). `game_version` names the build whose tables grade the
/// cards (absent = the default build, GH #877); an unsupported one is the
/// caller's 400.
pub fn reward_preview(body: &Value) -> Result<Value, String> {
    let version = crate::session::parse_game_version(body.get("game_version"))?;
    let list = |k: &str| -> Vec<Value> {
        body.get(k)
            .and_then(Value::as_array)
            .map(|a| a.iter().filter(|v| v.is_object()).cloned().collect())
            .unwrap_or_default()
    };
    Ok(deck::reward_preview_for(
        version,
        &list("deck"),
        &list("candidates"),
        deck::DEFAULT_SAMPLES,
        0,
    ))
}

/// `GET /api/live/deck-analysis` over a live state of build `version`.
#[must_use]
pub fn live_deck_analysis(state: &Value, version: sts2_core::GameVersion) -> Value {
    deck::analyze_deck_for(
        version,
        &deck_of(state),
        deck::DEFAULT_ENERGY,
        deck::DEFAULT_HAND,
        deck::DEFAULT_SAMPLES,
        0,
    )
}

/// `POST /api/wpa-review` over the recorded `wpa` records.
#[must_use]
pub fn wpa_review(store: &ReplayStore, config: &Value) -> Value {
    let rows: Vec<Value> = store
        .all()
        .iter()
        .filter(|s| crate::render::py::truthy(&s.wpa))
        .map(|s| {
            json!({
                "node_id": Value::Null,
                "idx": s.idx,
                "decision": s.decision,
                "floor": s.meta["floor"],
                "act": s.meta["act"],
                "value": s.wpa.get("value").cloned().unwrap_or(Value::Null),
                "taken_label": s.wpa.get("taken_label").cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    let won = store
        .all()
        .iter()
        .rev()
        .find(|s| s.decision == "game_over" && s.state.get("victory").is_some_and(|v| !v.is_null()))
        .map(|s| crate::render::py::truthy(&s.state["victory"]));
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
    runmode::wpa_review(
        &rows,
        top_k,
        min_swing,
        won,
        "this run log has no recorded value data — play it in the workbench (rust backend + PPO checkpoint) to record a value curve",
    )
}

// ── the engine-backed routes: replay the recorded line ───────────────────

/// A run rebuilt from the log's identity, stepped along its replay line.
#[derive(Debug)]
pub struct Rebuilt {
    game: Game,
    line: Vec<Value>,
    /// How many of `line` have been applied.
    applied: usize,
}

impl Rebuilt {
    /// Start the recorded run (never an action applied yet). `None` when
    /// the log cannot be replayed (no identity / no replayable line).
    pub fn start(store: &ReplayStore) -> Result<Self, String> {
        let ident = store
            .run_identity()
            .ok_or_else(|| "this run cannot be resumed (no replayable action line)".to_string())?;
        let last = store.len().saturating_sub(1);
        let line = store
            .replay_line(last)
            .ok_or_else(|| "this run cannot be resumed (no replayable action line)".to_string())?;
        let mut game = Game::new();
        game.game_version = ident.game_version;
        game.set_record_card_plays(true);
        game.set_record_combat_log(true);
        if ident.run_mode {
            let seed = ident.seed.parse::<i32>().map_err(|_| {
                format!(
                    "run-mode save carries a non-numeric seed {:?}; cannot restore it",
                    ident.seed
                )
            })?;
            if ident.e2e_setup {
                start_e2e(&mut game, seed, ident.ascension)?;
            } else {
                game.start_faithful(&ident.character, seed, ident.ascension)
                    .map_err(|GameError(m)| m)?;
            }
        } else {
            game.start(&ident.character, &ident.seed, ident.ascension, &ident.act1)
                .map_err(|GameError(m)| m)?;
        }
        Ok(Self {
            game,
            line,
            applied: 0,
        })
    }

    /// Apply recorded actions until `idx` of them have been applied.
    pub fn advance_to(&mut self, idx: usize) -> Result<(), String> {
        while self.applied < idx {
            let entry = self
                .line
                .get(self.applied)
                .ok_or_else(|| {
                    "this run cannot be resumed (no replayable action line)".to_string()
                })?
                .clone();
            let action = entry["action"].as_str().unwrap_or("");
            let args = entry.get("args").cloned().unwrap_or_else(|| json!({}));
            let total = self.line.len();
            let n = self.applied + 1;
            if action == NATIVE_ACTION {
                let result = self.game.take_native_value(&args).map_err(|e| {
                    format!("replay diverged at step {n}/{total} (action '{action}'): {e}")
                })?;
                if result != "advanced" && result != "run_over" {
                    return Err(format!(
                        "replay diverged at step {n}/{total} (action '{action}'): take_action returned '{result}'"
                    ));
                }
            } else {
                self.game.act(action, &args).map_err(|GameError(m)| {
                    format!("replay diverged at step {n}/{total} (action '{action}'): {m}")
                })?;
            }
            self.applied += 1;
        }
        Ok(())
    }

    /// The engine's combat state at the current step, if in a fight.
    #[must_use]
    pub fn combat(&self) -> Option<&CombatState> {
        self.game.combat_state()
    }
}

/// The recorded combat state at step `idx`, rebuilt by replay.
fn combat_at(store: &ReplayStore, idx: usize) -> Result<CombatState, String> {
    let mut r = Rebuilt::start(store)?;
    r.advance_to(idx)?;
    r.combat()
        .cloned()
        .ok_or_else(|| "step is not a combat decision".to_string())
}

/// `POST /api/recommend {idx, config}`: `reasoning.score` over the step's
/// engine state; the rows are cached back onto the step.
pub fn recommend(store: &mut ReplayStore, idx: i64, config: &Value) -> Result<Value, ApiError> {
    let raw_state = step_at(store, idx)?.state.clone();
    let echo = match config {
        Value::Object(_) => config.clone(),
        _ => json!({}),
    };
    let result = match combat_at(store, usize::try_from(idx).unwrap_or(0)) {
        Ok(cs) => advisors::score(&cs, &raw_state, config),
        Err(e) => json!({"error": e, "scores": [], "pv": Value::Null, "config": echo}),
    };
    if let Ok(s) = store.get_mut(idx) {
        s.scores = result
            .get("scores")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        s.pv = result.get("pv").cloned().unwrap_or(Value::Null);
    }
    Ok(result)
}

/// `POST /api/search_tree {idx, config}`: 409 off combat.
pub fn search_tree(store: &ReplayStore, idx: i64, config: &Value) -> Result<Value, ApiError> {
    let s = step_at(store, idx)?;
    if s.state.get("decision").and_then(Value::as_str) != Some("combat_play") {
        return Err(ApiError::new(
            409,
            "search tree only available for combat decisions",
        ));
    }
    let echo = match config {
        Value::Object(_) => config.clone(),
        _ => json!({}),
    };
    Ok(match combat_at(store, usize::try_from(idx).unwrap_or(0)) {
        Ok(cs) => advisors::search_tree(&cs, config),
        Err(e) => json!({"error": e, "nodes": [], "root": Value::Null, "pv": [], "config": echo}),
    })
}

/// Whether the review of one combat step must rebuild the run up to it: only
/// to search it (`need_search`: no recorded rows, or recompute) or to top up
/// the played move's recorded row. Everything else grades the recorded rows
/// as they stand, so the replay (and the state clone) is skipped.
fn needs_rebuild(need_search: bool, rows: &[Value], played: &Value) -> bool {
    need_search || advisors::played_needs_top_up(rows, played)
}

/// `POST /api/regret-review {config}`: every combat decision diffed against
/// the MCTS line — the recorded scores, or one forward replay re-searching
/// the steps without them (all of them under `recompute`).
#[must_use]
pub fn regret_review(store: &ReplayStore, config: &Value, loc: &'static Loc) -> Value {
    let mut cfg = match config {
        Value::Object(m) => m.clone(),
        _ => serde_json::Map::new(),
    };
    let recompute = cfg
        .remove("recompute")
        .is_some_and(|v| crate::render::py::truthy(&v));
    let blunder_pct = cfg
        .remove("blunder_pct")
        .and_then(|v| v.as_f64())
        .unwrap_or(advisors::DEFAULT_BLUNDER_DROP_PCT);
    let top_n = cfg
        .remove("top_n")
        .and_then(|v| v.as_i64())
        .map_or(2, |n| usize::try_from(n).unwrap_or(0));
    let mcts_cfg = Value::Object(cfg);
    let mut rebuilt: Option<Result<Rebuilt, String>> = None;
    let mut decisions = Vec::new();
    for s in store.all() {
        if s.state.get("decision").and_then(Value::as_str) != Some("combat_play") {
            continue;
        }
        let mut rows = if recompute {
            Vec::new()
        } else {
            s.scores.clone()
        };
        let params = advisors::params_for(&mcts_cfg, "mcts_recommend").ok();
        let need_search = rows.is_empty();
        let mut played_cs = None;
        let chosen = s.chosen_action.get("args").cloned().unwrap_or(Value::Null);
        if needs_rebuild(need_search, &rows, &chosen) {
            let r = rebuilt.get_or_insert_with(|| Rebuilt::start(store));
            if let Ok(r) = r.as_mut() {
                if r.advance_to(s.idx).is_ok() {
                    if let Some(cs) = r.combat() {
                        if need_search {
                            rows = advisors::score(
                                cs,
                                &resolve_state_names(Some(&s.state), loc),
                                &mcts_cfg,
                            )
                            .get("scores")
                            .and_then(Value::as_array)
                            .cloned()
                            .unwrap_or_default();
                        }
                        played_cs = Some(cs.clone());
                    }
                }
            }
        }
        // A hand-played move gets too few visits to grade at default sims, so
        // it would read "uncertain" and vanish from the review. Force a
        // gradable sample on it from the parent position — the same top-up the
        // live coach paths apply.
        if let (Some(cs), Some(params)) = (played_cs.as_ref(), params.as_ref()) {
            advisors::top_up_played(cs, &mut rows, &chosen, params);
        }
        // A recorded row carries its OWN label, so resolving the state is not
        // enough for it — a log written before the labels were resolved at
        // record time still holds raw keys here.
        for row in &mut rows {
            if let Some(name) = row.get("action").and_then(Value::as_str) {
                let resolved = crate::localization::resolve_name(Some(name), loc);
                if let Some(m) = row.as_object_mut() {
                    m.insert("action".to_string(), json!(resolved));
                }
            }
        }
        let step = DecisionStep {
            idx: i64::try_from(s.idx).unwrap_or(i64::MAX),
            node_id: None,
            // Resolve display names BEFORE anything reads the state. A run
            // saved by the live workbench persists the raw engine state, whose
            // card and potion names are localization keys, and the review
            // labels every row off `state["hand"]` — so an unresolved state
            // printed `played COLOSSUS.title instead of drink
            // COLORLESS_POTION.title`. Every other replay surface resolves on
            // the way out; this one scored the raw dict. Mirrors the live twin
            // `advisors::step_of`.
            state: resolve_state_names(Some(&s.state), loc),
            chosen: s.chosen_action.get("args").cloned().unwrap_or(Value::Null),
            turn: s.meta["turn"].clone(),
            floor: s.meta["floor"].clone(),
            act: s.meta["act"].clone(),
            // The replay review grades recorded rows; it names no AI line.
            best_line: Vec::new(),
        };
        if let Some(d) = advisors::assess_decision(&step, &rows, blunder_pct) {
            decisions.push(d);
        }
    }
    let mut out = advisors::summarize(decisions, top_n);
    out["config"] = json!({"blunder_pct": blunder_pct, "top_n": top_n});
    out
}

/// `resume_recording`'s checks for `POST /api/live/from_replay {idx}`:
/// the identity + line to rebuild through `Session::load_line`, or the
/// refusal (404 bad step, 409 unresumable).
pub fn resume_plan(
    store: &ReplayStore,
    idx: i64,
) -> Result<(crate::runlog::RunIdentity, Vec<Value>), ApiError> {
    step_at(store, idx)?;
    let cap = resume_capability(store);
    if cap["kind"].is_null() {
        return Err(ApiError::new(
            409,
            cap["reason"]
                .as_str()
                .unwrap_or("this recording cannot be resumed"),
        ));
    }
    let ident = store.run_identity();
    let line = store.replay_line(usize::try_from(idx).unwrap_or(0));
    match (ident, line) {
        (Some(i), Some(l)) => Ok((i, l)),
        _ => Err(ApiError::new(
            409,
            "this run cannot be resumed (no replayable action line)",
        )),
    }
}

#[cfg(test)]
mod rebuild_tests {
    use super::*;

    fn rows(best_visits: i64, played_visits: i64) -> Vec<Value> {
        vec![
            json!({"stable_idx": 0, "target_index": null, "visits": played_visits, "is_best": false}),
            json!({"stable_idx": 1, "target_index": null, "visits": best_visits, "is_best": true}),
        ]
    }

    fn play(card_index: i64) -> Value {
        json!({"action": "play", "card_index": card_index, "target_index": null})
    }

    #[test]
    fn rebuilds_when_a_search_is_needed() {
        assert!(needs_rebuild(true, &[], &play(0)));
        assert!(needs_rebuild(true, &rows(500, 120), &play(1)));
    }

    #[test]
    fn skips_the_rebuild_when_the_played_move_is_best() {
        assert!(!needs_rebuild(false, &rows(5, 2), &play(1)));
    }

    #[test]
    fn skips_the_rebuild_when_the_played_row_has_enough_visits() {
        const _: () = assert!(advisors::PLAYED_MIN_VISITS <= 120);
        assert!(!needs_rebuild(false, &rows(500, 120), &play(0)));
    }

    #[test]
    fn rebuilds_to_top_up_a_thin_played_row() {
        const _: () = assert!(advisors::PLAYED_MIN_VISITS > 12);
        assert!(needs_rebuild(false, &rows(500, 12), &play(0)));
    }

    #[test]
    fn skips_the_rebuild_when_the_played_row_is_missing() {
        assert!(!needs_rebuild(false, &rows(500, 12), &play(7)));
        assert!(!needs_rebuild(false, &rows(500, 12), &Value::Null));
    }
}

#[cfg(all(test, workbench_game_data))] // its one test asserts on real game text
mod tests {
    use super::*;

    /// A run saved by the live workbench persists the RAW engine state, where
    /// card names are localization keys. The regret review labels every row off
    /// `state["hand"]`, so an unresolved state printed
    /// `played STRIKE_IRONCLAD.title instead of DEFEND_IRONCLAD.title`.
    /// The recorded rows here carry the agent's own numbers, so no search runs.
    #[cfg(workbench_game_data)] // asserts on real game text (see build.rs)
    #[test]
    fn regret_review_resolves_localization_keys() {
        let text = [
            r#"{"step":0,"ts":"t","type":"meta","data":{"character":"Ironclad","seed":"S","ascension":0,"run_mode":true,"act1":"overgrowth","backend":"rust"}}"#,
            r#"{"step":1,"ts":"t","type":"state","data":{"decision":"combat_play","round":2,"hand":[{"index":0,"name":"STRIKE_IRONCLAD.title"},{"index":1,"name":"DEFEND_IRONCLAD.title"}],"player":{"potions":[]},"enemies":[]}}"#,
            r#"{"step":1,"ts":"t","type":"mcts","data":{"source":"agent","root_outcomes":[{"stable_idx":0,"action":"STRIKE_IRONCLAD.title","win_prob":0.4,"visits":100,"q":0.1,"is_best":false},{"stable_idx":1,"action":"DEFEND_IRONCLAD.title","win_prob":0.9,"visits":60,"q":0.4,"is_best":true}]}}"#,
            r#"{"step":1,"ts":"t","type":"action","data":{"cmd":"action","action":"play_card","args":{"action":"play","card_index":0,"target_index":null}}}"#,
            r#"{"step":2,"ts":"t","type":"state","data":{"decision":"map_select","floor":1,"act":1}}"#,
            r#"{"step":2,"ts":"t","type":"meta","data":{"summary":true,"final_won":false,"final_turn":2}}"#,
            "",
        ]
        .join("\n");
        let store = ReplayStore::parse("x.jsonl", &text).unwrap();
        let out = regret_review(&store, &json!({}), Loc::embedded());
        assert_eq!(out["summary"]["total_decisions"], json!(1));
        let b = &out["blunders"][0];
        assert_eq!(b["played_label"], json!("Strike"));
        assert_eq!(b["best_label"], json!("Defend"));
        assert_eq!(b["drop_pct"], json!(50));
        assert!(
            !b["message"].as_str().unwrap_or("").contains(".title"),
            "{b}"
        );
    }
}

/// `Game::start_e2e`, or an error in a build without the `e2e` feature.
fn start_e2e(game: &mut Game, seed: i32, ascension: u8) -> Result<(), String> {
    #[cfg(feature = "e2e")]
    {
        game.start_e2e(seed, ascension)
            .map(|_| ())
            .map_err(|GameError(m)| m)
    }
    #[cfg(not(feature = "e2e"))]
    {
        let _ = (game, seed, ascension);
        Err("this build has no e2e feature; cannot rebuild an e2e run".to_string())
    }
}
