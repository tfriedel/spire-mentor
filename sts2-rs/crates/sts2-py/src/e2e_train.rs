//! The e2e training env (`sts2_e2e::train::TrainEnv`) for the rollout worker.
//!
//! `ppo.e2e.rsenv.RsE2EEnv` wraps [`PyE2eTrainEnv`] behind `E2EEnv`'s
//! interface: every `reset` / `step` returns either a stored row (its
//! decision fields, the reward and, unless `encode=False`, the tokenized row
//! as bytes, see [`crate::e2e::tokrow_dict`]) or the segment's end. The
//! practice captures and the F9-sampled states queue up here until the host
//! drains them.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use pyo3::create_exception;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sts2_e2e::contracts as ct;
use sts2_e2e::tokens::{self, RareMap, TokCache};
use sts2_e2e::train::{Caps, SegmentSpec, StepError, TrainEnv, TrainStep, TrainTerminal};

create_exception!(
    sts2_rs,
    E2eEnvRaise,
    PyRuntimeError,
    "An error `E2eTrainEnv.reset` / `step` raises where the Python `E2EEnv`\n\
     raises instead of ending the segment: `args == (etype, msg)`, the Python\n\
     exception type's name and its message. `ppo.e2e.rsenv` re-raises it as\n\
     that type (`SchemaError`, `GameVersionMismatch`, ...)."
);

use crate::run_manager::PyRunManager;
use crate::value_to_py;

/// One training segment over its own run (`E2EEnv`'s Rust twin).
#[pyclass(name = "E2eTrainEnv", module = "sts2_rs", unsendable)]
#[derive(Debug)]
pub struct PyE2eTrainEnv {
    env: TrainEnv,
    cache: TokCache,
    encode: bool,
    priv_: bool,
    detail: bool,
    rare: Option<Arc<RareMap>>,
}

/// The combat teacher's `RareMap` (`RareMap.to_json()`), parsed once per
/// worker and shared by its envs.
#[pyclass(name = "E2eRareMap", module = "sts2_rs", frozen)]
#[derive(Debug)]
pub struct PyE2eRareMap {
    map: Arc<RareMap>,
}

#[pymethods]
impl PyE2eRareMap {
    /// From `RareMap.to_json()` (`{slot: [row, ...]}`).
    ///
    /// Raises:
    ///     ValueError: not a rare map.
    #[new]
    fn new(maps: &Bound<'_, PyAny>) -> PyResult<Self> {
        let v: serde_json::Value = pythonize::depythonize(maps)
            .map_err(|e| PyValueError::new_err(format!("E2eRareMap: {e}")))?;
        Ok(Self {
            map: Arc::new(RareMap::from_value(&v).map_err(PyValueError::new_err)?),
        })
    }
}

fn panic_msg(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "Rust panic".to_owned())
}

fn terminal_dict<'py>(py: Python<'py>, t: &TrainTerminal) -> PyResult<Bound<'py, PyDict>> {
    let o = PyDict::new(py);
    o.set_item("kind", "terminal")?;
    o.set_item("outcome", t.outcome)?;
    o.set_item("won", t.won)?;
    o.set_item("cap", t.cap)?;
    o.set_item("reward", t.reward)?;
    o.set_item("total_reward", t.total_reward)?;
    o.set_item("unassigned_reward", t.unassigned_reward)?;
    o.set_item("rows", t.rows)?;
    o.set_item("steps", t.steps)?;
    o.set_item("floor", t.floor)?;
    o.set_item("act", t.act)?;
    o.set_item("bosses", t.bosses)?;
    o.set_item("reached_boss2", t.reached_boss2)?;
    let c = PyDict::new(py);
    for (k, v) in &t.counts {
        c.set_item(k, v)?;
    }
    o.set_item("counts", c)?;
    o.set_item("etype", t.etype.as_deref())?;
    o.set_item("error", t.error.as_deref())?;
    Ok(o)
}

impl PyE2eTrainEnv {
    /// The pending row as the host reads it. The fields `RsDecision` never
    /// reads (members, classes, kinds, act, floor, phase, surface) cost a
    /// nested copy and Python lists per row; they are emitted only with
    /// `detail` (the lockstep check).
    fn decision_dict<'py>(
        &mut self,
        py: Python<'py>,
        reward: f64,
        bootstrap: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let d = self
            .env
            .decision()
            .ok_or_else(|| PyRuntimeError::new_err("a stored row without a pending decision"))?;
        let o = PyDict::new(py);
        o.set_item("kind", "decision")?;
        o.set_item("row", d.row)?;
        o.set_item("is_combat", d.is_combat())?;
        o.set_item("dtype", d.dtype)?;
        o.set_item("n_classes", d.n_classes())?;
        o.set_item("reward", reward)?;
        o.set_item("bootstrap", bootstrap)?;
        match &d.combat {
            Some(c) => o.set_item("unit", ("fight", c.fight))?,
            None => o.set_item("unit", ("act", d.act))?,
        }
        if self.detail {
            o.set_item("act", d.act)?;
            o.set_item("floor", d.floor)?;
            if let Some(c) = &d.combat {
                o.set_item("turn", c.turn)?;
            }
            o.set_item("members", &d.members)?;
            o.set_item("classes", &d.classes)?;
            o.set_item("kinds", &d.kinds)?;
            o.set_item("dtype_name", ct::dtype_name(d.dtype))?;
            if let Some(oo) = &d.ooc {
                o.set_item("phase", oo.phase)?;
                o.set_item("surface", oo.surface)?;
            }
        }
        if self.encode {
            // A tokenizer panic is a tokenizer error (`tok_error`, which the
            // worker meets as the row's `tok` raising), not an escaped
            // `PanicException`: the step's F9 states and captures are
            // already drained by then, and the segment ends as the Python
            // env ends it on a tokenizer assert.
            let (rm, cache, rare, priv_) = (
                self.env.rm(),
                &mut self.cache,
                self.rare.as_deref(),
                self.priv_,
            );
            let r = catch_unwind(AssertUnwindSafe(|| {
                tokens::encode_with(d, rm, cache, rare, priv_, true)
            }))
            .unwrap_or_else(|p| Err(format!("PanicException: {}", panic_msg(&*p))));
            match r {
                Ok(r) => o.set_item("tok", crate::e2e::tokrow_dict(py, &r)?)?,
                Err(e) => o.set_item("tok_error", e)?,
            }
        }
        Ok(o)
    }

    fn out<'py>(&mut self, py: Python<'py>, s: TrainStep) -> PyResult<Bound<'py, PyDict>> {
        match s {
            TrainStep::Decision { reward, bootstrap } => self.decision_dict(py, reward, bootstrap),
            TrainStep::Terminal(t) => terminal_dict(py, &t),
        }
    }

    /// Run `f` on the env, then the row or terminal as a dict. A panic once
    /// the segment has started ends it as `error/PanicException`; one
    /// before (inside `reset`'s first advance) raises, as `run.reset()`
    /// does outside `_drive`'s handler.
    fn guarded<'py>(
        &mut self,
        py: Python<'py>,
        f: impl FnOnce(&mut TrainEnv) -> Result<TrainStep, StepError>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let env = &mut self.env;
        let r = match catch_unwind(AssertUnwindSafe(|| f(env))) {
            Ok(r) => r,
            Err(p) if self.env.started() => Ok(self.env.fail("PanicException", panic_msg(&*p))),
            Err(p) => Err(StepError::Raise {
                etype: "PanicException",
                msg: panic_msg(&*p),
            }),
        };
        match r {
            Ok(s) => self.out(py, s),
            Err(StepError::Usage(m)) => Err(PyIndexError::new_err(m)),
            Err(StepError::Raise { etype, msg }) => Err(E2eEnvRaise::new_err((etype, msg))),
        }
    }
}

#[pymethods]
impl PyE2eTrainEnv {
    /// A segment over a copy of `rm`. `seed` is the run's seed; the F9
    /// sampler is `random.Random(f9_seed)` drawn once per applied action at
    /// `f9_rate`; `first_ooc` ends the segment at its first OOC row;
    /// `capture` queues a practice capture at every Move into an elite or
    /// boss node; `encode` tokenizes every row, with `priv_` its PRIV
    /// tokens and `rare` the combat teacher's map
    /// (`tokens.encode(d, rare=rare, priv=priv_)`). `caps` overrides
    /// `EnvConfig`'s caps (`{"run_steps", "fight_turns", "fight_decisions",
    /// "run_decisions"}`, each optional); `game_version` is the build every
    /// fight must simulate (`factory.check_fight`); `f9_queue_max` bounds the
    /// F9 states held between two drains (the rest count `f9/dropped`);
    /// `detail` adds the row fields only a check reads (`members`,
    /// `classes`, `kinds`, `act`, `floor`, `turn`, `phase`, `surface`).
    ///
    /// Raises:
    ///     ValueError: an unknown cap key or build.
    #[new]
    #[pyo3(signature = (rm, seed, *, first_ooc = false, capture = false, f9_seed = 0, f9_rate = 0.0, encode = true, priv_ = false, rare = None, caps = None, game_version = None, f9_queue_max = sts2_e2e::train::F9_QUEUE_MAX, detail = false))]
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    fn new(
        rm: &PyRunManager,
        seed: i64,
        first_ooc: bool,
        capture: bool,
        f9_seed: u64,
        f9_rate: f64,
        encode: bool,
        priv_: bool,
        rare: Option<&PyE2eRareMap>,
        caps: Option<std::collections::HashMap<String, i64>>,
        game_version: Option<&str>,
        f9_queue_max: usize,
        detail: bool,
    ) -> PyResult<Self> {
        let mut c = Caps::default();
        for (k, v) in caps.unwrap_or_default() {
            let bad = || PyValueError::new_err(format!("E2eTrainEnv: cap {k} = {v} out of range"));
            match k.as_str() {
                "run_steps" => c.run_steps = u64::try_from(v).map_err(|_| bad())?,
                "fight_turns" => c.fight_turns = v,
                "fight_decisions" => c.fight_decisions = usize::try_from(v).map_err(|_| bad())?,
                "run_decisions" => c.run_decisions = usize::try_from(v).map_err(|_| bad())?,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "E2eTrainEnv: unknown cap {k:?}"
                    )))
                }
            }
        }
        let game_version = game_version
            .map(|g| {
                g.parse()
                    .map_err(|e| PyValueError::new_err(format!("E2eTrainEnv: {e}")))
            })
            .transpose()?;
        Ok(Self {
            env: TrainEnv::new(
                rm.inner.clone(),
                seed,
                SegmentSpec {
                    first_ooc,
                    capture,
                    f9_seed,
                    f9_rate,
                    caps: c,
                    f9_queue_max,
                    game_version,
                },
            ),
            cache: TokCache::new(),
            encode,
            priv_,
            detail,
            rare: rare.map(|r| Arc::clone(&r.map)),
        })
    }

    /// Advance to the first stored row.
    ///
    /// Raises:
    ///     E2eEnvRaise: the advance to the first output failed, or a fight
    ///         simulates another build (the Python env raises both).
    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.guarded(py, TrainEnv::reset)
    }

    /// Apply merged class `choice` of the pending row.
    ///
    /// Raises:
    ///     IndexError: no pending row, a bootstrap row, `choice` out of range.
    ///     E2eEnvRaise: a fight simulates another build.
    fn step<'py>(&mut self, py: Python<'py>, choice: usize) -> PyResult<Bound<'py, PyDict>> {
        self.guarded(py, |e| e.step(choice))
    }

    /// Apply member `ai` of the pending row.
    ///
    /// Raises:
    ///     IndexError, E2eEnvRaise: as `step`.
    fn step_action<'py>(&mut self, py: Python<'py>, ai: usize) -> PyResult<Bound<'py, PyDict>> {
        self.guarded(py, |e| e.step_action(ai))
    }

    /// A copy of the live run.
    fn rm(&self) -> PyRunManager {
        PyRunManager::from_inner(self.env.rm().clone())
    }

    /// The segment ended.
    #[getter]
    fn done(&self) -> bool {
        self.env.done()
    }

    /// Stored rows so far.
    #[getter]
    fn rows(&self) -> usize {
        self.env.rows()
    }

    /// The action log so far.
    #[getter]
    fn log_idx(&self) -> Vec<i64> {
        self.env.log_idx().to_vec()
    }

    /// The env's counters so far (without `forced/<type>`).
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let c = PyDict::new(py);
        for (k, v) in self.env.counts() {
            c.set_item(k, v)?;
        }
        Ok(c)
    }

    /// The segment's end (None before it).
    fn terminal<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.env
            .terminal()
            .map(|t| terminal_dict(py, t))
            .transpose()
    }

    /// The captures since the last drain: `[{cell, snap, move, steps, rows,
    /// idx}]` (`snap` the run before the Move).
    fn drain_captures<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for c in self.env.drain_captures() {
            let o = PyDict::new(py);
            o.set_item("cell", c.cell)?;
            o.set_item("snap", PyRunManager::from_inner(c.snap))?;
            o.set_item("move", value_to_py(py, &c.action)?)?;
            o.set_item("steps", c.steps)?;
            o.set_item("rows", c.rows)?;
            o.set_item("idx", c.idx)?;
            out.append(o)?;
        }
        Ok(out)
    }

    /// The F9-sampled post-action runs since the last drain, in order.
    fn drain_f9(&mut self) -> Vec<PyRunManager> {
        self.env
            .drain_f9()
            .into_iter()
            .map(PyRunManager::from_inner)
            .collect()
    }

    /// Drop the tokenizer's per-run cache (a new run on the same env).
    fn begin_run(&mut self) {
        self.cache.begin_run();
    }

    fn __repr__(&self) -> String {
        format!(
            "E2eTrainEnv(seed={}, rows={}, done={})",
            self.env.seed(),
            self.env.rows(),
            self.env.done()
        )
    }
}
