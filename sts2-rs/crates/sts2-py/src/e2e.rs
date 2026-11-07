//! Instinct — the e2e Ironclad student (`sts2-e2e`) — for Python.
//!
//! The game mod's advisor (`realgame_advisor.py` → `realgame_instinct.py`)
//! keeps an engine [`RunManager`] in step with the live run and asks the
//! student what it would play there. The workbench does the same over its
//! own run (`sts2-workbench/src/session_e2e.rs`); this module is the same
//! three pieces, no more:
//!
//! - [`PyE2eStudent`]: the net (`.e2ew` weights), checked against the build
//!   it was trained on.
//! - [`PyE2eDriver`]: the student's per-run state — the env driver
//!   ([`Driver`]: fight tracking and the fight's intent log, `grant_picked`)
//!   and the tokenizer's per-run cache ([`TokCache`]).
//! - [`PyE2ePeek`]: what the env does at the run's current position
//!   ([`Driver::peek`]): a decision with its merged classes, a forced step,
//!   or the end.
//!
//! Everything a Python caller sees is plain data (dicts of the engine's
//! run-action JSON), so the advisor maps a class to the live screen without
//! knowing the student's internals. The tokenizer, the forward and the env
//! driver are the parity-gated Rust port (`sts2-e2e/tests/*_parity.rs`);
//! nothing here re-implements any of them.

use std::sync::Arc;
use std::time::Instant;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde_json::{json, Value};

use sts2_e2e::contracts as ct;
use sts2_e2e::env::{ApplyKind, Decision, Driver, Next};
use sts2_e2e::net::E2eNet;
use sts2_e2e::tokens::{self, TokCache};

use crate::run_manager::PyRunManager;
use crate::value_to_py;

/// The only build the student plays (`sts2_e2e::env::GAME_VERSION`).
pub const GAME_VERSION: &str = sts2_e2e::env::GAME_VERSION;

/// Softmax in f64 (max-subtracted), as the workbench's `e2e::softmax`.
fn softmax(logits: &[f32]) -> Vec<f64> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = logits
        .iter()
        .map(|&x| (f64::from(x) - f64::from(m)).exp())
        .collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|x| x / z).collect()
}

fn apply_kind_name(kind: ApplyKind) -> &'static str {
    match kind {
        ApplyKind::Combat => "combat",
        ApplyKind::Run { .. } => "run",
    }
}

/// A decision as plain data: its type, its merged classes (each with the
/// run-action dict of every member, the representative first) and the
/// selection it is part of.
fn decision_value(d: &Decision) -> Value {
    let classes: Vec<Value> = d
        .members
        .iter()
        .enumerate()
        .map(|(ci, members)| {
            let actions: Vec<Value> = members
                .iter()
                .map(|&ai| d.actions.get(ai).cloned().unwrap_or(Value::Null))
                .collect();
            let obs_actions: Vec<Value> = members
                .iter()
                .map(|&ai| d.obs_actions.get(ai).cloned().unwrap_or(Value::Null))
                .collect();
            let kind = members
                .first()
                .and_then(|&ai| d.kinds.get(ai))
                .map_or("", |&k| ct::kind_name(k));
            json!({
                "class": ci,
                "kind": kind,
                "actions": actions,
                "obs_actions": obs_actions,
            })
        })
        .collect();
    json!({
        "row": d.row,
        "kind": d.kind.as_str(),
        "decision_type": ct::dtype_name(d.dtype),
        "is_combat": d.is_combat(),
        "n_classes": d.n_classes(),
        "classes": classes,
        "selection": d.sel.to_value(),
        "event_name": d.event_name,
        "act": d.act,
        "floor": d.floor,
        "turn": d.turn(),
        "surface": d.ooc.as_ref().and_then(|o| o.surface),
        "phase": d.ooc.as_ref().map(|o| o.phase),
    })
}

/// The student's weights. Immutable after load; shared across threads.
#[pyclass(name = "E2eStudent", module = "sts2_rs", frozen)]
pub struct PyE2eStudent {
    net: Arc<E2eNet>,
}

impl std::fmt::Debug for PyE2eStudent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("E2eStudent")
            .field("variant", &self.net.header().variant)
            .finish_non_exhaustive()
    }
}

#[pymethods]
impl PyE2eStudent {
    /// Load the `.e2ew` weights at `path`.
    ///
    /// Raises:
    ///     ValueError: a malformed file, or one stamped for another game
    ///         build than the student plays.
    ///     OSError: the file cannot be read.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path)?;
        let net = E2eNet::from_bytes(&bytes).map_err(PyValueError::new_err)?;
        if let Some(v) = net.header().game_version.as_deref() {
            if v != GAME_VERSION {
                return Err(PyValueError::new_err(format!(
                    "e2e weights are stamped for {v}, the student plays {GAME_VERSION}"
                )));
            }
        }
        Ok(Self { net: Arc::new(net) })
    }

    /// The build the student plays.
    #[getter]
    #[allow(clippy::unused_self)]
    fn game_version(&self) -> &'static str {
        GAME_VERSION
    }

    /// The weights file's provenance stamps (`source`, `variant`, hashes).
    #[getter]
    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let h = self.net.header();
        value_to_py(
            py,
            &json!({
                "format": h.format,
                "source": h.source,
                "source_sha256": h.source_sha256,
                "schema_sha256": h.schema_sha256,
                "vocab_sha256": h.vocab_sha256,
                "game_version": h.game_version,
                "variant": h.variant,
            }),
        )
    }

    /// Tokenize `peek`'s decision at `rm` (with `driver`'s per-run cache)
    /// and run the student: `{class, probs, ms_tokenize, ms_net}`, `class`
    /// being the greedy merged class and `probs` the softmax per class.
    ///
    /// `rm` must still be at the position `peek` was taken at.
    ///
    /// Raises:
    ///     ValueError: `peek` is not a decision.
    ///     RuntimeError: the tokenizer or the forward refused the decision.
    fn decide<'py>(
        &self,
        py: Python<'py>,
        driver: &mut PyE2eDriver,
        peek: &PyE2ePeek,
        rm: &PyRunManager,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Next::Decision(d) = &peek.next else {
            return Err(PyValueError::new_err("decide: the peek is not a decision"));
        };
        let t0 = Instant::now();
        let row =
            tokens::encode(d, &rm.inner, &mut driver.cache).map_err(PyRuntimeError::new_err)?;
        let t1 = Instant::now();
        let logits = self
            .net
            .try_logits(&row.into())
            .map_err(PyRuntimeError::new_err)?;
        let t2 = Instant::now();
        value_to_py(
            py,
            &json!({
                "class": sts2_e2e::net::argmax(&logits),
                "probs": softmax(&logits),
                "logits": logits,
                "ms_tokenize": (t1 - t0).as_secs_f64() * 1000.0,
                "ms_net": (t2 - t1).as_secs_f64() * 1000.0,
            }),
        )
    }
}

/// What the env does at one run position ([`Driver::peek`]).
#[pyclass(name = "E2ePeek", module = "sts2_rs", frozen)]
#[derive(Debug)]
pub struct PyE2ePeek {
    next: Next,
}

#[pymethods]
impl PyE2ePeek {
    /// `"decision"`, `"forced"` or `"terminal"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.next {
            Next::Decision(_) => "decision",
            Next::Forced(_) => "forced",
            Next::Terminal(_) => "terminal",
        }
    }

    /// The decision as a dict (`decision_type`, `classes[]` with each
    /// member's run-action dict, `selection`, ...), else `None`.
    #[getter]
    fn decision<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.next {
            Next::Decision(d) => value_to_py(py, &decision_value(d)).map(Some),
            _ => Ok(None),
        }
    }

    /// A forced step: `{action, kind}` (`kind` `"run"` / `"combat"`), else
    /// `None`.
    #[getter]
    fn forced<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.next {
            Next::Forced(f) => value_to_py(
                py,
                &json!({"action": f.action, "kind": apply_kind_name(f.kind)}),
            )
            .map(Some),
            _ => Ok(None),
        }
    }

    /// The end: `{outcome, won, cap, error, floor, act}`, else `None`.
    #[getter]
    fn terminal<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.next {
            Next::Terminal(t) => value_to_py(
                py,
                &json!({
                    "outcome": t.outcome,
                    "won": t.won,
                    "cap": t.cap,
                    "error": t.error,
                    "floor": t.floor,
                    "act": t.act,
                }),
            )
            .map(Some),
            _ => Ok(None),
        }
    }

    /// The run-action dict the env applies for merged class `class` (its
    /// representative) and how: `{action, kind}`.
    ///
    /// Raises:
    ///     ValueError: not a decision, or `class` out of range.
    fn class_action<'py>(&self, py: Python<'py>, class: usize) -> PyResult<Bound<'py, PyAny>> {
        let Next::Decision(d) = &self.next else {
            return Err(PyValueError::new_err(
                "class_action: the peek is not a decision",
            ));
        };
        let (action, kind) = d.class_action(class).map_err(PyValueError::new_err)?;
        value_to_py(
            py,
            &json!({"action": action, "kind": apply_kind_name(kind)}),
        )
    }

    fn __repr__(&self) -> String {
        match &self.next {
            Next::Decision(d) => format!(
                "E2ePeek(decision {} with {} classes)",
                ct::dtype_name(d.dtype),
                d.n_classes()
            ),
            Next::Forced(f) => format!("E2ePeek(forced {})", f.action),
            Next::Terminal(t) => format!("E2ePeek(terminal {})", t.outcome),
        }
    }
}

/// The student's per-run state: the env driver and the tokenizer cache.
///
/// One per live run. `peek` is not idempotent at a decision (it counts the
/// row; the caps read the counts): ask "what would the student play here?"
/// on a [`Self::copy`] unless the answer is what the run will do.
#[pyclass(name = "E2eDriver", module = "sts2_rs")]
#[derive(Clone, Debug)]
pub struct PyE2eDriver {
    driver: Driver,
    cache: TokCache,
}

#[pymethods]
impl PyE2eDriver {
    /// A fresh state for the run with integer seed `seed` (a run start, or
    /// the first look at a run already under way).
    #[new]
    fn new(seed: i64) -> Self {
        Self {
            driver: Driver::new(seed),
            cache: TokCache::new(),
        }
    }

    /// An independent copy (the cache shares its immutable map read).
    fn copy(&self) -> Self {
        self.clone()
    }

    /// What the env does at `rm`'s current position. Binds a pending
    /// event's callback (idempotent); nothing else touches the run.
    fn peek(&mut self, rm: &mut PyRunManager) -> PyE2ePeek {
        PyE2ePeek {
            next: self.driver.peek(&mut rm.inner),
        }
    }

    /// Show the driver the current combat position, so the fight's intent
    /// log (a policy input) sees every round's first snapshot. A no-op out
    /// of combat.
    ///
    /// Raises:
    ///     RuntimeError: the combat state could not be emitted.
    fn observe_combat_position(&mut self, rm: &PyRunManager) -> PyResult<()> {
        self.driver
            .observe_combat_position(&rm.inner)
            .map_err(PyRuntimeError::new_err)
    }

    /// Picks already made on the open card grant.
    #[getter]
    fn grant_picked(&self) -> i64 {
        self.driver.grant_picked()
    }

    #[setter]
    fn set_grant_picked(&mut self, n: i64) {
        self.driver.set_grant_picked(n);
    }

    /// Apply `action` (a run-action dict; `kind` `"run"` or `"combat"`, as
    /// a peek names it) to `rm` with the env's bookkeeping. Returns the
    /// engine's result string.
    ///
    /// Raises:
    ///     ValueError: an unknown `kind`, or the engine/parser refused it.
    #[pyo3(signature = (rm, action, kind = "run", strict = true))]
    fn apply(
        &mut self,
        rm: &mut PyRunManager,
        action: &Bound<'_, PyAny>,
        kind: &str,
        strict: bool,
    ) -> PyResult<&'static str> {
        let action: Value = pythonize::depythonize(action)
            .map_err(|e| PyValueError::new_err(format!("apply: bad action: {e}")))?;
        let kind = match kind {
            "run" => ApplyKind::Run { strict },
            "combat" => ApplyKind::Combat,
            other => {
                return Err(PyValueError::new_err(format!(
                    "apply: unknown kind {other:?} (expected \"run\" or \"combat\")"
                )))
            }
        };
        self.driver
            .apply(&mut rm.inner, &action, kind)
            .map_err(PyValueError::new_err)
    }

    /// Start the tokenizer cache over (a new run on the same state).
    fn begin_run(&mut self) {
        self.cache.begin_run();
    }
}

/// The campaign run the student trained on for `seed` (Ironclad, v0.111.0,
/// Neow start, rolled act 1, agent mode, interactive potion swap), at
/// `ascension` (default: the training one). For tests and parity checks.
#[pyfunction]
#[pyo3(signature = (seed, ascension = None))]
pub fn e2e_new_run(seed: i64, ascension: Option<u8>) -> PyResult<PyRunManager> {
    if !(0..1_i64 << 31).contains(&seed) {
        return Err(PyValueError::new_err(format!(
            "seed {seed} outside [0, 2**31)"
        )));
    }
    let rm = sts2_e2e::env::new_run_at(seed, ascension.unwrap_or(sts2_e2e::env::ASCENSION));
    Ok(PyRunManager::from_inner(rm))
}

/// The student plays `seed` greedily from the start on a fresh campaign run
/// (at most `max_decisions` decisions) — the `examples/play.rs` loop, for a
/// Python-side check that the binding plays exactly the native env.
/// `{won, outcome, decisions, chosen, error, floor, act}`.
#[pyfunction]
#[pyo3(signature = (student, seed, max_decisions = 6000))]
pub fn e2e_selfplay<'py>(
    py: Python<'py>,
    student: &PyE2eStudent,
    seed: i64,
    max_decisions: usize,
) -> PyResult<Bound<'py, PyAny>> {
    use sts2_e2e::env::{E2eEnv, Step};
    if !(0..1_i64 << 31).contains(&seed) {
        return Err(PyValueError::new_err(format!(
            "seed {seed} outside [0, 2**31)"
        )));
    }
    let net = Arc::clone(&student.net);
    let out = py.allow_threads(move || {
        let mut env = E2eEnv::over_run(sts2_e2e::env::new_run(seed), seed);
        let mut cache = TokCache::new();
        let mut chosen: Vec<usize> = Vec::new();
        let mut step = env.reset();
        loop {
            match step {
                Step::Terminal(t) => {
                    break json!({
                        "won": t.won, "outcome": t.outcome, "error": t.error,
                        "floor": t.floor, "act": t.act,
                        "decisions": chosen.len(), "chosen": chosen,
                    })
                }
                Step::Decision(d) => {
                    if chosen.len() >= max_decisions {
                        break json!({"won": false, "outcome": "stopped", "error": null,
                                     "decisions": chosen.len(), "chosen": chosen});
                    }
                    let class = match tokens::encode(&d, env.rm(), &mut cache)
                        .and_then(|row| net.try_logits(&row.into()))
                    {
                        Ok(logits) => sts2_e2e::net::argmax(&logits),
                        Err(e) => {
                            break json!({"won": false, "outcome": "error", "error": e,
                                         "decisions": chosen.len(), "chosen": chosen})
                        }
                    };
                    chosen.push(class);
                    step = env.step(class);
                }
            }
        }
    });
    value_to_py(py, &out)
}
