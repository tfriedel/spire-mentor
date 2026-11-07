//! wasm-bindgen shell over [`sts2_workbench`].
//!
//! Everything platform-specific for the browser lives here: the JS-facing
//! constructor, the `call(method, route, body)` entry point the Web Worker
//! drives, the save payload builders and the session persistence
//! import/export. The logic itself is in `sts2-workbench`, which the
//! golden-parity tests exercise natively.
//!
//! Every method returns a JSON **string** of the shape
//! `{"status": <http status>, "body": <json>}` so the worker can wrap it in
//! a `Response` without a second parse of the body, and so an error is a
//! plain value rather than a thrown `JsError` (the SPA's `json<T>` reads
//! the status off the `Response`).
//!
//! `std::env::var_os` reads inside `sts2-core` (map generator / pruning
//! knobs, event-log echo, RNG trace flags) all return `None` on
//! wasm32-unknown-unknown, which is the production default for each.

use serde_json::{json, Value};
use sts2_workbench::api::{ApiError, Config};
use wasm_bindgen::prelude::*;

/// Crate version, so the worker can log which engine build it loaded.
#[wasm_bindgen]
#[must_use]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// `Date.now()`: the millisecond clock the `wpa-verify` deadline reads
/// (`Instant` does not exist on wasm32-unknown-unknown).
fn js_now_ms() -> f64 {
    js_sys::Date::now()
}

/// `performance.now()` (the worker's high-resolution clock) for the e2e
/// brain's per-decision timing; `Date.now()` where it is missing.
fn js_perf_ms() -> f64 {
    use wasm_bindgen::JsCast;
    let global = js_sys::global();
    js_sys::Reflect::get(&global, &JsValue::from_str("performance"))
        .ok()
        .filter(|p| !p.is_undefined() && !p.is_null())
        .and_then(|perf| {
            let now = js_sys::Reflect::get(&perf, &JsValue::from_str("now")).ok()?;
            let now: js_sys::Function = now.dyn_into().ok()?;
            now.call0(&perf).ok()?.as_f64()
        })
        .unwrap_or_else(js_sys::Date::now)
}

/// `[[filename, text], …]` → the `(filename, text)` pairs the log routes take.
fn parse_logs(logs_json: &str) -> Result<Vec<(String, String)>, ApiError> {
    let v: Value = serde_json::from_str(logs_json)
        .map_err(|e| ApiError::new(400, format!("invalid logs JSON: {e}")))?;
    Ok(v.as_array()
        .into_iter()
        .flatten()
        .filter_map(|pair| {
            let name = pair.get(0)?.as_str()?;
            let text = pair.get(1)?.as_str()?;
            Some((name.to_string(), text.to_string()))
        })
        .collect())
}

fn reply(result: Result<Value, ApiError>) -> String {
    let (status, body) = match result {
        Ok(body) => (200_u16, body),
        Err(e) => (e.status, e.body()),
    };
    json!({"status": status, "body": body}).to_string()
}

fn parse_body(body_json: &str) -> Result<Value, ApiError> {
    if body_json.trim().is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(body_json)
        .map_err(|e| ApiError::new(400, format!("invalid JSON body: {e}")))
}

/// The workbench, one per worker.
#[derive(Debug)]
#[wasm_bindgen]
pub struct Workbench {
    inner: sts2_workbench::api::Workbench,
}

#[wasm_bindgen]
impl Workbench {
    /// `config_json`: `{"assets_base": "https://…"}` (defaults to `/assets`).
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<Workbench, JsError> {
        console_error_panic_hook::set_once();
        let cfg: Value = if config_json.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str(config_json)
                .map_err(|e| JsError::new(&format!("invalid config JSON: {e}")))?
        };
        let mut config = Config::default();
        if let Some(base) = cfg.get("assets_base").and_then(Value::as_str) {
            config.assets_base = base.to_string();
        }
        config.now_ms = Some(js_now_ms);
        config.perf_ms = Some(js_perf_ms);
        Ok(Self {
            inner: sts2_workbench::api::Workbench::new(config),
        })
    }

    /// Dispatch one `/api/*` route. `method` is `GET` or `POST`.
    #[must_use]
    pub fn call(&mut self, method: &str, route: &str, body_json: &str) -> String {
        reply(parse_body(body_json).and_then(|body| self.inner.call(method, route, &body)))
    }

    /// The streaming advisor routes (`…/card-advisor/stream`,
    /// `…/decision-advisor/stream`): `on_frame` receives each NDJSON frame
    /// as a JSON **string** (`{"type": "start"|"tick"|"result"|"error", …}`),
    /// in order, the last one terminal; the worker posts each to the page,
    /// which turns them into the response body. The return value is the
    /// usual reply envelope with status 200 and a null body — the frames ARE
    /// the response. See `sts2_workbench::api::Workbench::call_stream`.
    #[must_use]
    pub fn call_stream(
        &mut self,
        method: &str,
        route: &str,
        body_json: &str,
        on_frame: &js_sys::Function,
    ) -> String {
        let this = JsValue::NULL;
        let body = match parse_body(body_json) {
            Ok(b) => b,
            Err(e) => {
                let frame = json!({"type": "error", "error": e.detail}).to_string();
                let _ = on_frame.call1(&this, &JsValue::from_str(&frame));
                return reply(Ok(Value::Null));
            }
        };
        self.inner.call_stream(method, route, &body, &mut |frame| {
            let _ = on_frame.call1(&this, &JsValue::from_str(&frame.to_string()));
        });
        reply(Ok(Value::Null))
    }

    /// `POST /api/live/save` minus the storage: the payload to persist.
    /// `saved_at` is the host clock as ISO-8601 (`2026-09-02T10:00:00Z`).
    #[must_use]
    pub fn build_save(&self, name: &str, saved_at: &str) -> String {
        reply(self.inner.build_save(name, saved_at))
    }

    /// `POST /api/saves/load`: restore a stored payload; the body is the
    /// enriched step. `now` (ISO-8601, the host's clock) stamps the run log
    /// of the live run the restore replaces (`take_run_log`).
    #[must_use]
    pub fn restore_save(&mut self, payload_json: &str, now: &str) -> String {
        reply(parse_body(payload_json).and_then(|p| self.inner.restore_save(&p, now)))
    }

    /// The list-view header of a stored payload (`GET /api/saves` rows).
    #[must_use]
    pub fn save_header(payload_json: &str) -> String {
        reply(parse_body(payload_json).map(|p| sts2_workbench::api::Workbench::save_header(&p)))
    }

    /// The live session's tree without node states (`null` when no run).
    #[must_use]
    pub fn export_persist(&self) -> String {
        self.inner.export_persist().to_string()
    }

    /// Rebuild the session from `export_persist` output. `progress` is
    /// called with `(done, total)` while the active path replays. The body
    /// is the enriched active step.
    #[must_use]
    pub fn import_persist(&mut self, payload_json: &str, progress: &js_sys::Function) -> String {
        let this = JsValue::NULL;
        reply(parse_body(payload_json).and_then(|p| {
            self.inner.import_persist(&p, |done, total| {
                let _ = progress.call2(
                    &this,
                    &JsValue::from_f64(u32::try_from(done).map_or(f64::MAX, f64::from)),
                    &JsValue::from_f64(u32::try_from(total).map_or(f64::MAX, f64::from)),
                );
            })
        }))
    }

    /// `POST /api/load {path}` with the stored log text: parse it into the
    /// replay store; `{status, body: {count, resume}}`.
    #[must_use]
    pub fn load_run(&mut self, path: &str, text: &str) -> String {
        reply(self.inner.load_run(path, text))
    }

    /// The run log the last `/api/live/quit` wrote, as JSON
    /// `{"filename", "text"}`, or `null` — collected once by the page,
    /// which stores it under `filename`.
    #[must_use]
    pub fn take_run_log(&mut self) -> String {
        match self.inner.take_run_log() {
            Some(log) => json!({"filename": log.filename, "text": log.text}).to_string(),
            None => "null".to_string(),
        }
    }

    /// `GET /api/mcts-runs` over `logs_json` = `[[filename, text], …]`.
    #[must_use]
    pub fn mcts_runs(logs_json: &str) -> String {
        reply(parse_logs(logs_json).map(|logs| sts2_workbench::api::Workbench::mcts_runs(&logs)))
    }

    /// `GET /api/mcts-runs/aggregate` over `logs_json`.
    #[must_use]
    pub fn aggregate_runs(logs_json: &str) -> String {
        reply(
            parse_logs(logs_json).map(|logs| sts2_workbench::api::Workbench::aggregate_runs(&logs)),
        )
    }

    /// The champion ONNX filename the page should fetch for `character`
    /// (`/ppo/<file>`), as JSON `{"status", "body": "<file>"}` (400 for an
    /// unknown character).
    #[must_use]
    pub fn ppo_champion_file(character: &str) -> String {
        reply(sts2_workbench::api::Workbench::ppo_champion_file(character).map(|f| json!(f)))
    }

    /// Load `character`'s champion from the fetched ONNX bytes; the live
    /// run picks it up if it is that character's. 400 with tract's error
    /// text when the bytes are not a loadable export.
    #[must_use]
    pub fn load_ppo(&mut self, character: &str, bytes: &[u8]) -> String {
        reply(self.inner.load_ppo(character, bytes))
    }

    /// Record why the page could not fetch `character`'s champion (offline,
    /// a 404 …) so the panel and the brain can say so.
    pub fn set_ppo_error(&mut self, character: &str, reason: &str) {
        self.inner.set_ppo_error(character, reason);
    }

    /// Whether `character`'s champion is loaded.
    #[must_use]
    pub fn has_ppo(&self, character: &str) -> bool {
        self.inner.has_ppo(character)
    }

    /// The e2e student's weights filename the page fetches on demand
    /// (`/e2e/<file>`).
    #[must_use]
    pub fn e2e_file() -> String {
        sts2_workbench::api::Workbench::e2e_file().to_string()
    }

    /// Load the e2e student from the fetched `.e2ew` bytes; the live run
    /// picks it up. 400 when the bytes are not a weights file of the build
    /// the student plays.
    #[must_use]
    pub fn load_e2e(&mut self, bytes: &[u8]) -> String {
        reply(self.inner.load_e2e(bytes))
    }

    /// Record why the page could not fetch the e2e weights.
    pub fn set_e2e_error(&mut self, reason: &str) {
        self.inner.set_e2e_error(reason);
    }

    /// Whether the e2e student is loaded.
    #[must_use]
    pub fn has_e2e(&self) -> bool {
        self.inner.has_e2e()
    }

    /// Whether a run is live (cheap probe for the worker).
    #[must_use]
    pub fn has_live_run(&self) -> bool {
        self.inner.session().is_some()
    }

    // ── SimLab sweeps (GH #707) ────────────────────────────────────────
    // A sweep is a host-stepped job: the worker calls `sweep_tick` between
    // the page's requests until `active` is false, persisting each id in
    // `finished` through `export_sweep`, and hands stored sweeps back with
    // `import_sweep` on the first listing after a reload.

    /// Play up to `max_trials` trials of the oldest running sweep; JSON
    /// `{"active": bool, "finished": [ids]}`.
    #[must_use]
    pub fn sweep_tick(&mut self, max_trials: u32) -> String {
        self.inner.sweep_tick(max_trials as usize).to_string()
    }

    /// Whether a sweep still has trials to play.
    #[must_use]
    pub fn has_running_sweep(&self) -> bool {
        self.inner.has_running_sweep()
    }

    /// A sweep as the host stores it (`"null"` for an unknown id).
    #[must_use]
    pub fn export_sweep(&self, id: &str) -> String {
        self.inner
            .export_sweep(id)
            .map_or_else(|| "null".to_string(), |v| v.to_string())
    }

    /// Restore a stored sweep; the usual reply envelope.
    #[must_use]
    pub fn import_sweep(&mut self, doc_json: &str) -> String {
        reply(parse_body(doc_json).and_then(|d| self.inner.import_sweep(&d)))
    }
}
