//! `sts2_rs.AdvisorMirrorCore`: the live advisor's per-fight mirror
//! (`sts2_advisor::advisor_mirror`, GH #1426 step 4).
//! `realgame_advisor_mirror.AdvisorMirror` keeps its API, writes the log
//! lines and divergences this queues, and hands out the `live` / fork views.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;
use sts2_advisor::advisor_mirror::{AdvisorMirror, Level, On};
use sts2_advisor::card_select::Arity;
use sts2_advisor::search::RecommendConfig;

use crate::combat_mirror::{desyncs_to_py, err, json, PyDesync};
use crate::PyCombatState;

fn on(which: &str) -> PyResult<On> {
    match which {
        "live" => Ok(On::Live),
        "fork" => Ok(On::Fork),
        other => Err(PyValueError::new_err(format!("no mirror {other:?}"))),
    }
}

/// A Python exception as `(class name, message)`.
pub(crate) fn py_exc(py: Python<'_>, e: &PyErr) -> (String, String) {
    let name = e
        .get_type(py)
        .name()
        .map_or_else(|_| "Exception".to_string(), |n| n.to_string());
    (name, e.value(py).to_string())
}

/// A board with no engine fight, as `combat_wire_from_bridge` raised it.
pub(crate) fn board_err(e: sts2_advisor::board::BoardError) -> PyErr {
    use sts2_advisor::board::BoardError;
    match e {
        BoardError::Unsupported(m) => crate::state_dict::UnsupportedStateError::new_err(m),
        BoardError::Overflow(m) => pyo3::exceptions::PyOverflowError::new_err(m),
        BoardError::GameVersion(m) => PyValueError::new_err(m),
    }
}

/// A log level as the `logging` method name.
pub(crate) const fn level_name(level: Level) -> &'static str {
    match level {
        Level::Debug => "debug",
        Level::Info => "info",
        Level::Warning => "warning",
    }
}

/// `wire_for(combat, character=, ascension=, combat_id=, seed=)`, the
/// Python board import a card-select answer runs, taking the GIL; its
/// exception as `(class name, message)`.
pub(crate) fn call_wire_for(
    wire_for: &Py<PyAny>,
    combat: &Value,
    character: Option<&str>,
    ascension: i64,
    combat_id: i64,
    seed: u64,
) -> Result<Value, (String, String)> {
    Python::with_gil(|py| {
        let call = || -> PyResult<Value> {
            let kw = pyo3::types::PyDict::new(py);
            kw.set_item("character", character)?;
            kw.set_item("ascension", ascension)?;
            kw.set_item("combat_id", combat_id)?;
            kw.set_item("seed", seed)?;
            let combat = crate::value_to_py(py, combat)?;
            let wire = wire_for.bind(py).call((combat,), Some(&kw))?;
            json(&wire, "wire")
        };
        call().map_err(|e| py_exc(py, &e))
    })
}

/// The advisor's mirror. Python's `realgame_advisor_mirror.AdvisorMirror`
/// wraps it.
#[pyclass(name = "AdvisorMirrorCore", module = "sts2_rs")]
#[derive(Debug)]
pub struct PyAdvisorMirror {
    pub(crate) core: AdvisorMirror,
}

#[pymethods]
impl PyAdvisorMirror {
    #[new]
    #[pyo3(signature = (enabled = true, damage_trace = false))]
    fn new(enabled: bool, damage_trace: bool) -> Self {
        Self {
            core: AdvisorMirror::new(enabled, damage_trace),
        }
    }

    #[getter]
    const fn enabled(&self) -> bool {
        self.core.enabled
    }

    #[setter]
    const fn set_enabled(&mut self, v: bool) {
        self.core.enabled = v;
    }

    #[getter]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::value_to_py(py, &self.core.stats())
    }

    #[getter]
    fn declines(&self) -> std::collections::BTreeMap<String, i64> {
        self.core.declines.clone()
    }

    #[getter]
    fn desync_paths(&self) -> std::collections::BTreeMap<String, i64> {
        self.core.desync_paths.clone()
    }

    #[getter]
    fn adopted_paths(&self) -> std::collections::BTreeMap<String, i64> {
        self.core.adopted_paths.clone()
    }

    // -- the live mirror and the fork, for the Python views ----------------

    fn alive(&mut self, which: &str) -> PyResult<bool> {
        Ok(self.core.mirror_mut(on(which)?).is_some_and(|m| m.alive()))
    }

    /// The mirror's engine state (a copy), or `None`.
    fn state(&mut self, which: &str) -> PyResult<Option<PyCombatState>> {
        Ok(self
            .core
            .mirror_mut(on(which)?)
            .and_then(|m| m.mirrored())
            .map(|m| PyCombatState::from_mirrored(m.state.clone(), m.seed, m.intent_damage_known)))
    }

    fn state_generation(&mut self, which: &str) -> PyResult<Option<u64>> {
        Ok(self
            .core
            .mirror_mut(on(which)?)
            .map(|m| m.state_generation()))
    }

    fn advance(
        &mut self,
        py: Python<'_>,
        which: &str,
        action: &Bound<'_, PyAny>,
    ) -> PyResult<&'static str> {
        let action = json(action, "action")?;
        let m = self
            .core
            .mirror_mut(on(which)?)
            .ok_or_else(|| err(py, sts2_advisor::combat_mirror::MirrorError::NoState))?;
        m.advance(&action).map_err(|e| err(py, e))
    }

    fn reconcile<'py>(
        &mut self,
        py: Python<'py>,
        which: &str,
        wire: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<PyDesync<'py>>> {
        let wire = json(wire, "wire")?;
        let m = self
            .core
            .mirror_mut(on(which)?)
            .ok_or_else(|| err(py, sts2_advisor::combat_mirror::MirrorError::NoState))?;
        let desyncs = m.reconcile(&wire).map_err(|e| err(py, e))?;
        desyncs_to_py(py, &desyncs)
    }

    /// One honest search on a clone of the mirror; off the GIL.
    fn search<'py>(
        &mut self,
        py: Python<'py>,
        which: &str,
        cfg: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = RecommendConfig::from_value(&json(cfg, "cfg")?);
        let m = self
            .core
            .mirror_mut(on(which)?)
            .ok_or_else(|| err(py, sts2_advisor::combat_mirror::MirrorError::NoState))?;
        let out = py
            .allow_threads(|| m.search(&cfg))
            .map_err(|e| err(py, e))?;
        crate::value_to_py(py, &out)
    }

    #[getter]
    const fn generation(&self) -> u64 {
        self.core.generation()
    }

    fn tainted(&mut self, which: &str) -> PyResult<bool> {
        Ok(self.core.mirror_mut(on(which)?).is_some_and(|m| m.tainted))
    }

    #[allow(clippy::too_many_arguments)]
    fn inject_choice<'py>(
        &mut self,
        py: Python<'py>,
        which: &str,
        kind: &str,
        options: &Bound<'py, PyAny>,
        min_select: i64,
        max_select: i64,
        can_skip: bool,
        origin: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Vec<PyDesync<'py>>> {
        let Value::Array(options) = json(options, "options")? else {
            return Err(PyValueError::new_err("options must be a list"));
        };
        let origin = origin
            .map(|o| json(o, "origin"))
            .transpose()?
            .unwrap_or(Value::Null);
        let m = self
            .core
            .mirror_mut(on(which)?)
            .ok_or_else(|| err(py, sts2_advisor::combat_mirror::MirrorError::NoState))?;
        let graded = m
            .inject_choice(kind, options, min_select, max_select, can_skip, origin)
            .map_err(|e| err(py, e))?;
        desyncs_to_py(py, &graded)
    }

    fn advance_through_choice(
        &mut self,
        py: Python<'_>,
        which: &str,
        selection: Vec<i64>,
    ) -> PyResult<&'static str> {
        let m = self
            .core
            .mirror_mut(on(which)?)
            .ok_or_else(|| err(py, sts2_advisor::combat_mirror::MirrorError::NoState))?;
        m.advance_through_choice(&selection).map_err(|e| err(py, e))
    }

    fn fork_live(&mut self, label: &str) -> bool {
        self.core.fork_live(label)
    }

    fn drop_mirror(&mut self, which: &str, reason: &str) -> PyResult<()> {
        match on(which)? {
            On::Live => self.core.drop_fight(reason),
            On::Fork => self.core.drop_fork(reason),
        }
        Ok(())
    }

    // -- AdvisorMirror -----------------------------------------------------

    fn decline(&mut self, reason: &str) {
        self.core.decline(reason);
    }

    fn drop(&mut self, reason: &str) {
        self.core.drop_fight(reason);
    }

    fn on_rewind(&mut self, kind: &Bound<'_, PyAny>) -> PyResult<()> {
        self.core.on_rewind(&json(kind, "kind")?);
        Ok(())
    }

    fn on_combat_board(
        &mut self,
        wire: &Bound<'_, PyAny>,
        state: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        Ok(self
            .core
            .on_combat_board(&json(wire, "wire")?, &json(state, "state")?))
    }

    fn on_world_action(&mut self, msg: &Bound<'_, PyAny>) -> PyResult<()> {
        self.core.on_world_action(&json(msg, "msg")?);
        Ok(())
    }

    fn source_is_replayable(
        &mut self,
        source_model: &Bound<'_, PyAny>,
        source_kind: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        Ok(self.core.source_is_replayable(
            &json(source_model, "source_model")?,
            &json(source_kind, "source_kind")?,
        ))
    }

    fn replay_opening_drink(
        &mut self,
        which: &str,
        source_model: &Bound<'_, PyAny>,
        world_wire: &Bound<'_, PyAny>,
        target_index: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        Ok(self.core.replay_opening_drink(
            on(which)?,
            &json(source_model, "source_model")?,
            &json(world_wire, "world_wire")?,
            &json(target_index, "target_index")?,
        ))
    }

    fn replay_end_turn(&mut self, which: &str, world_wire: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self
            .core
            .replay_end_turn(on(which)?, &json(world_wire, "world_wire")?))
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_opening_play(
        &mut self,
        which: &str,
        source_model: &Bound<'_, PyAny>,
        world_wire: &Bound<'_, PyAny>,
        source_kind: &Bound<'_, PyAny>,
        target_index: &Bound<'_, PyAny>,
        upgraded: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        Ok(self.core.replay_opening_play(
            on(which)?,
            &json(source_model, "source_model")?,
            &json(world_wire, "world_wire")?,
            &json(source_kind, "source_kind")?,
            &json(target_index, "target_index")?,
            &json(upgraded, "upgraded")?,
        ))
    }

    fn remember_picks(
        &mut self,
        source_model: &Bound<'_, PyAny>,
        offered: &Bound<'_, PyAny>,
        picks: Vec<i64>,
    ) -> PyResult<()> {
        let offered = json(offered, "offered")?;
        self.core.remember_picks(
            &json(source_model, "source_model")?,
            offered.as_array().map_or(&[], Vec::as_slice),
            &picks,
        );
        Ok(())
    }

    fn next_remembered_pick(
        &mut self,
        source_model: &Bound<'_, PyAny>,
        offered: &Bound<'_, PyAny>,
    ) -> PyResult<Option<usize>> {
        let offered = json(offered, "offered")?;
        Ok(self.core.next_remembered_pick(
            &json(source_model, "source_model")?,
            offered.as_array().map_or(&[], Vec::as_slice),
        ))
    }

    /// `answer_card_select`. `picker(state, n_options, min_select,
    /// max_select)` picks on the parked fork's state instead of the combat
    /// MCTS; `wire_for(combat, character, ascension, combat_id, seed)` is
    /// the board import. The work runs off the GIL; both callbacks take it.
    #[pyo3(signature = (state, cfg, sims, wire_for, picker = None, picker_note = None))]
    #[allow(clippy::too_many_arguments)]
    fn answer_card_select<'py>(
        &mut self,
        py: Python<'py>,
        state: &Bound<'py, PyAny>,
        cfg: Option<&Bound<'py, PyAny>>,
        sims: Option<u32>,
        wire_for: Py<PyAny>,
        picker: Option<Py<PyAny>>,
        picker_note: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = json(state, "state")?;
        let cfg = cfg
            .map(|c| json(c, "cfg").map(|v| RecommendConfig::from_value(&v)))
            .transpose()?;
        let core = &mut self.core;
        let out = py.allow_threads(|| {
            let mut wire_for = |combat: &Value,
                                character: Option<&str>,
                                ascension: i64,
                                combat_id: i64,
                                seed: u64| {
                call_wire_for(&wire_for, combat, character, ascension, combat_id, seed)
            };
            let mut pick = |s: &sts2_core::CombatState, a: Arity| {
                Python::with_gil(|py| {
                    let call = || -> PyResult<Vec<i64>> {
                        let state = Py::new(py, PyCombatState::from_mirrored(s.clone(), 0, true))?;
                        let f = picker.as_ref().expect("only called when given");
                        f.bind(py)
                            .call1((state, a.n_options, a.min_select, a.max_select))?
                            .extract()
                    };
                    call().map_err(|e| py_exc(py, &e))
                })
            };
            let picker_ref: Option<&mut sts2_advisor::card_select::StatePicker<'_>> =
                if picker.is_some() {
                    Some(&mut pick)
                } else {
                    None
                };
            core.answer_card_select(
                &state,
                cfg.as_ref(),
                sims,
                picker_ref,
                picker_note.as_deref(),
                &mut wire_for,
            )
        });
        match out {
            Some(v) => crate::value_to_py(py, &v),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn note(&self) -> String {
        self.core.note()
    }

    /// The queued log lines as `(level, message)`, oldest first.
    fn take_log(&mut self) -> Vec<(&'static str, String)> {
        self.core
            .take_log()
            .into_iter()
            .map(|(level, msg)| (level_name(level), msg))
            .collect()
    }

    /// The divergences every mirror recorded, as `(paths, report)`.
    fn take_divergences<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Vec<(Vec<String>, Bound<'py, PyAny>)>> {
        self.core
            .take_divergences()
            .into_iter()
            .map(|d| Ok((d.paths, crate::value_to_py(py, &d.report)?)))
            .collect()
    }
}

/// `realgame_advisor_mirror._enemy_hp`: the living enemies' HP, in order.
#[pyfunction]
fn advisor_enemy_hp<'py>(py: Python<'py>, wire: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let hp = sts2_advisor::advisor_mirror::enemy_hp(&json(wire, "wire")?);
    crate::value_to_py(py, &Value::Array(hp))
}

/// `realgame_advisor.compute_combat_advice`
/// (`sts2_advisor::advise_combat`): `(advice, fight, character, ascension,
/// rows, log)`. `cfg` is `dataclasses.asdict(RecommendConfig)`; the
/// decision's config is `cfg` with that character and ascension, `sims`,
/// and `c_puct=None`. `log` is what the advisor logs, as
/// `(level, message)`; the mirror's own lines stay queued on it.
/// `plan_budget_s` caps the turn plan's re-searches (infinite: no cap, as
/// a replay wants).
#[pyfunction]
#[pyo3(signature = (state, cfg, sims, names, mirror = None, plan_budget_s = sts2_advisor::turn_plan::PLAN_BUDGET_S))]
#[allow(clippy::type_complexity)]
fn advice_compute_combat<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyAny>,
    cfg: &Bound<'py, PyAny>,
    sims: u32,
    names: &crate::display_names::PyDisplayNames,
    mirror: Option<&Bound<'py, PyAdvisorMirror>>,
    plan_budget_s: f64,
) -> PyResult<(
    Bound<'py, PyAny>,
    PyCombatState,
    Option<String>,
    i64,
    Bound<'py, PyAny>,
    Vec<(&'static str, String)>,
)> {
    use sts2_advisor::advise_combat::{compute_combat_advice, AdviseError};
    let state = json(state, "state")?;
    let cfg = RecommendConfig::from_value(&json(cfg, "cfg")?);
    let mut held = mirror.map(Bound::borrow_mut);
    let core = held.as_deref_mut().map(|c| &mut c.core);
    let names = &names.names;
    let mut log = Vec::new();
    let out = py.allow_threads(|| {
        compute_combat_advice(&state, &cfg, sims, plan_budget_s, core, names, &mut log)
    });
    drop(held);
    let log = log
        .into_iter()
        .map(|(level, msg)| (level_name(level), msg))
        .collect();
    let out = out.map_err(|e| match e {
        AdviseError::Headless(e) => PyValueError::new_err(e.to_string()),
        AdviseError::Board(e) => board_err(e),
        AdviseError::State(e) => err(py, e),
        AdviseError::Search(m) => PyValueError::new_err(m),
        AdviseError::Unmapped(m) => pyo3::exceptions::PyKeyError::new_err(m),
    })?;
    let fight = PyCombatState::from_mirrored(
        out.fight.state,
        out.fight.seed,
        out.fight.intent_damage_known,
    );
    Ok((
        crate::value_to_py(py, &out.advice)?,
        fight,
        out.cfg.character,
        out.cfg.ascension,
        crate::value_to_py(py, &Value::Array(out.rows))?,
        log,
    ))
}

/// The advice step of `advice_compute_combat` over a given search `result`
/// (the fight a fresh rebuild of `state`): `(advice, rows)`. `research`,
/// called with a `CombatState` and returning its line, continues the turn
/// plan past the PV; `None` stops it there. For checking how a search
/// result becomes advice without running one.
#[pyfunction]
#[pyo3(signature = (state, result, cfg, sims, names, research = None))]
fn advice_from_result<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyAny>,
    result: &Bound<'py, PyAny>,
    cfg: &Bound<'py, PyAny>,
    sims: u32,
    names: &crate::display_names::PyDisplayNames,
    research: Option<Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    use sts2_advisor::advise_combat::{self as ac, AdviseError};

    let state = json(state, "state")?;
    let call_cfg = ac::decision_cfg(
        &state,
        &RecommendConfig::from_value(&json(cfg, "cfg")?),
        sims,
    );
    let headless = sts2_advisor::headless::to_headless_combat_play(&state)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (wire, _) = sts2_advisor::board::combat_wire(
        &state,
        call_cfg.character.as_deref(),
        call_cfg.ascension,
        call_cfg.combat_id,
        call_cfg.seed,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let fight = sts2_advisor::combat_mirror::build_state(&wire).map_err(|e| err(py, e))?;
    let mut call = |at: &sts2_core::CombatState| -> Option<Vec<Value>> {
        let research = research.as_ref()?;
        let obj = PyCombatState::from_mirrored(at.clone(), fight.seed, fight.intent_damage_known);
        let line = research.call1((Py::new(py, obj).ok()?,)).ok()?;
        match pythonize::depythonize::<Value>(&line).ok()? {
            Value::Array(a) => Some(a),
            _ => None,
        }
    };
    let research: Option<&mut dyn FnMut(&sts2_core::CombatState) -> Option<Vec<Value>>> =
        if research.as_ref().is_some_and(|r| !r.is_none()) {
            Some(&mut call)
        } else {
            None
        };
    let (advice, rows) = ac::advice_from_result(
        &state,
        &wire,
        &headless,
        &fight.state,
        &json(result, "result")?,
        sims,
        &names.names,
        research,
        sts2_advisor::turn_plan::PLAN_BUDGET_S,
    )
    .map_err(|e| match e {
        AdviseError::Unmapped(m) => pyo3::exceptions::PyKeyError::new_err(m),
        other => PyValueError::new_err(other.to_string()),
    })?;
    Ok((
        crate::value_to_py(py, &advice)?,
        crate::value_to_py(py, &Value::Array(rows))?,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAdvisorMirror>()?;
    m.add_function(wrap_pyfunction!(advice_from_result, m)?)?;
    m.add_function(wrap_pyfunction!(advice_compute_combat, m)?)?;
    m.add_function(wrap_pyfunction!(advisor_enemy_hp, m)?)?;
    Ok(())
}
