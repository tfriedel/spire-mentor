//! `sts2_rs.CombatMirrorCore`: the combat mirror's per-fight state machine
//! (`sts2_advisor::combat_mirror`, GH #1426 step 1).
//!
//! `combat_mirror.CombatMirror` keeps its API, the search, the divergence
//! log and the `on_divergence` hook, and delegates the rest here. A
//! divergence under STRICT raises [`MirrorDivergedError`] with
//! `(message, report)`; the Python class drains [`PyCombatMirror::take_divergences`]
//! after every call (log, hook) and re-raises it as `MirrorDivergence`.

use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde_json::{Map, Value};
use sts2_advisor::combat_mirror::{CombatMirror, MirrorError, Policy};
use sts2_advisor::mirror::Desync;
use sts2_advisor::search::RecommendConfig;

use crate::PyCombatState;

create_exception!(
    sts2_rs,
    MirrorDivergedError,
    PyRuntimeError,
    "A combat-mirror divergence under the STRICT policy. `args` is \
     `(message, report)`; `report` is `None` when no fight was mirrored."
);

pub(crate) fn json(arg: &Bound<'_, PyAny>, name: &str) -> PyResult<Value> {
    pythonize::depythonize(arg)
        .map_err(|e| PyValueError::new_err(format!("{name} is not JSON-shaped: {e}")))
}

pub(crate) fn err(py: Python<'_>, e: MirrorError) -> PyErr {
    match e {
        MirrorError::Diverged(d) => match crate::value_to_py(py, &d.report) {
            Ok(report) => MirrorDivergedError::new_err((d.message, report.unbind())),
            Err(e) => e,
        },
        MirrorError::NoState => {
            MirrorDivergedError::new_err((MirrorError::NoState.to_string(), py.None()))
        }
        MirrorError::Wire(e) => crate::state_dict::wire_err(e),
        MirrorError::Invariant(m) => crate::state_dict::UnsupportedStateError::new_err(m),
        MirrorError::Value(m) => PyValueError::new_err(m),
        MirrorError::Key(m) => pyo3::exceptions::PyKeyError::new_err(m),
    }
}

/// A desync as Python sees it: `(path, mirror, world)`.
pub(crate) type PyDesync<'py> = (String, Bound<'py, PyAny>, Bound<'py, PyAny>);

pub(crate) fn desyncs_to_py<'py>(
    py: Python<'py>,
    desyncs: &[Desync],
) -> PyResult<Vec<PyDesync<'py>>> {
    desyncs
        .iter()
        .map(|d| {
            Ok((
                d.path.clone(),
                crate::value_to_py(py, &d.mirror)?,
                crate::value_to_py(py, &d.world)?,
            ))
        })
        .collect()
}

fn opt_to_py<'py>(py: Python<'py>, v: Option<&Value>) -> PyResult<Bound<'py, PyAny>> {
    v.map_or_else(
        || Ok(py.None().into_bound(py)),
        |v| crate::value_to_py(py, v),
    )
}

/// The mirror. Python's `combat_mirror.CombatMirror` wraps it.
#[pyclass(name = "CombatMirrorCore", module = "sts2_rs")]
#[derive(Debug)]
pub struct PyCombatMirror {
    core: CombatMirror,
    /// The `CombatState` last handed out by `state`, and the
    /// generation it copies: re-reading an unchanged mirror returns the same
    /// object, as the Python attribute did.
    cached: Option<(u64, Py<PyCombatState>)>,
}

impl PyCombatMirror {
    const fn wrap(core: CombatMirror) -> Self {
        Self { core, cached: None }
    }
}

#[pymethods]
impl PyCombatMirror {
    #[new]
    #[pyo3(signature = (policy = "strict", label = "", damage_trace = false))]
    fn new(policy: &str, label: &str, damage_trace: bool) -> PyResult<Self> {
        let policy = Policy::from_name(policy)
            .ok_or_else(|| PyValueError::new_err(format!("unknown mirror policy {policy:?}")))?;
        Ok(Self::wrap(CombatMirror::new(policy, label, damage_trace)))
    }

    #[getter]
    const fn alive(&self) -> bool {
        self.core.alive()
    }

    #[getter]
    const fn policy(&self) -> &'static str {
        self.core.policy.name()
    }

    #[getter]
    fn label(&self) -> &str {
        &self.core.label
    }

    #[getter]
    const fn tainted(&self) -> bool {
        self.core.tainted
    }

    #[setter]
    const fn set_tainted(&mut self, v: bool) {
        self.core.tainted = v;
    }

    #[getter]
    fn drop_reason(&self) -> &str {
        &self.core.drop_reason
    }

    #[getter]
    const fn teammates(&self) -> u32 {
        self.core.teammates
    }

    #[setter]
    const fn set_teammates(&mut self, v: u32) {
        self.core.teammates = v;
    }

    #[getter]
    const fn crossed_shuffle(&self) -> bool {
        self.core.crossed_shuffle
    }

    #[setter]
    const fn set_crossed_shuffle(&mut self, v: bool) {
        self.core.crossed_shuffle = v;
    }

    /// The options a just-answered choice picked, armed for one reconcile.
    #[getter]
    fn choice_in_flight<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        opt_to_py(
            py,
            self.core
                .choice_in_flight
                .clone()
                .map(Value::Array)
                .as_ref(),
        )
    }

    #[setter]
    fn set_choice_in_flight(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.core.choice_in_flight = if v.is_none() {
            None
        } else {
            match json(v, "choice_in_flight")? {
                Value::Array(items) => Some(items),
                _ => return Err(PyValueError::new_err("choice_in_flight must be a list")),
            }
        };
        Ok(())
    }

    /// The `MirrorStats` counters, as `MirrorStats.as_dict()`.
    #[getter]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::value_to_py(py, &self.core.stats.as_value())
    }

    #[getter]
    fn last_adopted<'py>(&self, py: Python<'py>) -> PyResult<Vec<PyDesync<'py>>> {
        desyncs_to_py(py, &self.core.last_adopted)
    }

    /// Bumped whenever the call's wire became [`Self::wire`].
    #[getter]
    const fn wire_generation(&self) -> u64 {
        self.core.wire_generation()
    }

    /// The enemy uid renames its rebuilds made since the last call
    /// (`[(uid then, uid now)]`), or `None` (GH #1651).
    fn take_enemy_remap(&mut self) -> Option<Vec<(u32, u32)>> {
        self.core.take_enemy_remap()
    }

    /// The last wire reality handed the mirror (its JSON copy).
    #[getter]
    fn wire<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        opt_to_py(py, self.core.wire())
    }

    /// The live engine state, or `None`. A copy: read it, do not advance it.
    #[getter]
    fn state(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyCombatState>>> {
        let Some(m) = self.core.mirrored() else {
            self.cached = None;
            return Ok(None);
        };
        let generation = self.core.state_generation();
        if let Some((g, obj)) = &self.cached {
            if *g == generation {
                return Ok(Some(obj.clone_ref(py)));
            }
        }
        let obj = Py::new(
            py,
            PyCombatState::from_mirrored(m.state.clone(), m.seed, m.intent_damage_known),
        )?;
        self.cached = Some((generation, obj.clone_ref(py)));
        Ok(Some(obj))
    }

    fn parked_choice<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        opt_to_py(py, self.core.parked_choice())
    }

    fn seed(&mut self, py: Python<'_>, wire: &Bound<'_, PyAny>) -> PyResult<()> {
        let wire = json(wire, "wire")?;
        self.core.seed(&wire).map_err(|e| err(py, e))
    }

    #[pyo3(signature = (label = ""))]
    fn fork(&self, label: &str) -> Option<Self> {
        self.core.fork(label).map(Self::wrap)
    }

    #[pyo3(signature = (reason = ""))]
    fn drop(&mut self, reason: &str) {
        self.core.drop_fight(reason);
    }

    fn reconcile<'py>(
        &mut self,
        py: Python<'py>,
        wire: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<PyDesync<'py>>> {
        let wire = json(wire, "wire")?;
        let desyncs = self.core.reconcile(&wire).map_err(|e| err(py, e))?;
        desyncs_to_py(py, &desyncs)
    }

    fn adopt_unseen_play(&mut self, py: Python<'_>, wire: &Bound<'_, PyAny>) -> PyResult<bool> {
        let wire = json(wire, "wire")?;
        self.core.adopt_unseen_play(&wire).map_err(|e| err(py, e))
    }

    fn advance(&mut self, py: Python<'_>, action: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        let action = json(action, "action")?;
        self.core.advance(&action).map_err(|e| err(py, e))
    }

    #[pyo3(signature = (kind, options, min_select, max_select, can_skip = false, origin = None))]
    #[allow(clippy::too_many_arguments)]
    fn inject_choice<'py>(
        &mut self,
        py: Python<'py>,
        kind: &str,
        options: Vec<Bound<'py, PyAny>>,
        min_select: i64,
        max_select: i64,
        can_skip: bool,
        origin: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Vec<PyDesync<'py>>> {
        let options = options
            .iter()
            .map(|o| json(o, "option"))
            .collect::<PyResult<Vec<_>>>()?;
        let origin = origin.map_or(Ok(Value::Null), |o| json(o, "origin"))?;
        let graded = self
            .core
            .inject_choice(kind, options, min_select, max_select, can_skip, origin)
            .map_err(|e| err(py, e))?;
        desyncs_to_py(py, &graded)
    }

    fn arm_turn_start_resume(&mut self, py: Python<'_>, after_relic: &str) -> PyResult<bool> {
        self.core
            .arm_turn_start_resume(after_relic)
            .map_err(|e| err(py, e))
    }

    fn arm_turn_start_resume_after_power(
        &mut self,
        py: Python<'_>,
        after_power: &str,
    ) -> PyResult<bool> {
        self.core
            .arm_turn_start_resume_after_power(after_power)
            .map_err(|e| err(py, e))
    }

    fn advance_through_choice(
        &mut self,
        py: Python<'_>,
        selection: Vec<i64>,
    ) -> PyResult<&'static str> {
        self.core
            .advance_through_choice(&selection)
            .map_err(|e| err(py, e))
    }

    /// Record a divergence the host found (the search's belief-leak guard).
    #[pyo3(signature = (message, kind, paths, extra = None))]
    fn diverge(
        &mut self,
        py: Python<'_>,
        message: String,
        kind: &str,
        paths: Vec<String>,
        extra: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let extra = match extra.map(|e| json(e, "extra")).transpose()? {
            None => Map::new(),
            Some(Value::Object(m)) => m,
            Some(_) => return Err(PyValueError::new_err("extra must be a dict")),
        };
        let wire = self.core.wire().cloned();
        self.core
            .diverge(message, kind, wire, paths, None, extra)
            .map_err(|e| err(py, e))
    }

    /// `CombatMirror.search`: one honest search on a clone of the mirror
    /// (`cfg`: the `RecommendConfig` as a dict). The search runs off the GIL.
    fn search<'py>(
        &mut self,
        py: Python<'py>,
        cfg: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = RecommendConfig::from_value(&json(cfg, "cfg")?);
        let core = &mut self.core;
        let out = py
            .allow_threads(|| core.search(&cfg))
            .map_err(|e| err(py, e))?;
        crate::value_to_py(py, &out)
    }

    /// `CombatMirror.search_choice_picks`: the open choice's picks, one
    /// search per pick, on a private clone.
    fn search_choice_picks(
        &mut self,
        py: Python<'_>,
        cfg: &Bound<'_, PyAny>,
        hand: &Bound<'_, PyAny>,
        n_options: i64,
        min_select: i64,
        max_select: i64,
    ) -> PyResult<Vec<i64>> {
        let cfg = RecommendConfig::from_value(&json(cfg, "cfg")?);
        let hand = json(hand, "hand")?;
        let hand = hand.as_array().cloned().unwrap_or_default();
        let core = &mut self.core;
        py.allow_threads(|| {
            core.search_choice_picks(&cfg, &hand, n_options, min_select, max_select)
        })
        .map_err(|e| err(py, e))
    }

    /// The first half of `realgame_mcts.pick_card_select_mirrored`: reconcile
    /// against the screen's `wire` (its `headless` form beside it), recognize
    /// the screen, inject the world's options, arm a blocked turn start.
    /// `(n_options, min_select, max_select)`.
    fn prepare_card_select(
        &mut self,
        py: Python<'_>,
        state: &Bound<'_, PyAny>,
        wire: &Bound<'_, PyAny>,
        headless: &Bound<'_, PyAny>,
    ) -> PyResult<(i64, i64, i64)> {
        let a = sts2_advisor::card_select::prepare_card_select(
            &mut self.core,
            &json(state, "state")?,
            &json(wire, "wire")?,
            &json(headless, "headless")?,
        )
        .map_err(|e| err(py, e))?;
        Ok((a.n_options, a.min_select, a.max_select))
    }

    /// Count one search.
    const fn note_search(&mut self) {
        self.core.note_search();
    }

    /// The divergences recorded since the last call, oldest first, as
    /// `(paths, report)`.
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCombatMirror>()?;
    m.add(
        "MirrorDivergedError",
        m.py().get_type::<MirrorDivergedError>(),
    )?;
    Ok(())
}
