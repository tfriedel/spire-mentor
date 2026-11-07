//! Named saves — the payload shape of `sts2-pycli/python/workbench/saves.py`.
//!
//! A save captures the **active line** of a session: the run identity
//! `(character, seed, ascension, act1, backend, run_mode, game_version)` plus the ordered
//! `(action, args)` pairs from the root to the active leaf. Loading replays
//! that line, so a save is only valid against the engine that recorded it.
//!
//! Storage is the host's business (IndexedDB in the browser); this module
//! only builds and reads the JSON.

use serde_json::{json, Map, Value};
use sts2_core::GameVersion;

use crate::session::Progress;

/// Save files written by this module carry this version; readers skip
/// mismatches.
pub const SCHEMA_VERSION: u64 = 1;

/// Backend assumed for saves written before the `backend` field existed.
pub const DEFAULT_BACKEND: &str = "csharp";

/// `act1` value for saves written before the field existed (GH #544).
pub const DEFAULT_ACT1: &str = "overgrowth";

/// `slugify`: lowercase, runs of non-alphanumerics → `-`, trimmed; `save`
/// if nothing remains.
#[must_use]
pub fn slugify(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut pending_dash = false;
    for c in name.trim().to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            if pending_dash && !out.is_empty() {
                out.push('-');
            }
            pending_dash = false;
            out.push(c);
        } else {
            pending_dash = true;
        }
    }
    if out.is_empty() {
        "save".to_string()
    } else {
        out
    }
}

/// `write_save`'s payload.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn build_save(
    name: &str,
    saved_at: &str,
    character: &str,
    seed: &str,
    ascension: u8,
    backend: &str,
    run_mode: bool,
    act1: &str,
    game_version: GameVersion,
    progress: &Progress,
    line: &[Value],
    node_values: &[Option<f64>],
) -> Value {
    let mut out = json!({
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "saved_at": saved_at,
        "character": character,
        "seed": seed,
        "ascension": ascension,
        "backend": backend,
        "run_mode": run_mode,
        "act1": act1,
        "progress": progress.to_value(),
        "step_count": line.len(),
        "line": line,
        "node_values": node_values,
    });
    // GH #801: written only for a non-default build (a v0.107.1 save keeps
    // its bytes); `RunIdentity::from_save` reads absence as v0.107.1.
    if !game_version.is_default() {
        out["game_version"] = json!(game_version.as_str());
    }
    out
}

/// `_header`: the list-view metadata (no `line` / `node_values`).
#[must_use]
pub fn header(payload: &Value) -> Value {
    let mut out = Map::new();
    if let Some(m) = payload.as_object() {
        for (k, v) in m {
            if k != "line" && k != "node_values" {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    Value::Object(out)
}

/// Whether a payload is a save this module can read.
#[must_use]
pub fn is_valid(payload: &Value) -> bool {
    payload.is_object()
        && payload.get("schema_version").and_then(Value::as_u64) == Some(SCHEMA_VERSION)
}

/// `RunIdentity::from_save`: what rebuilds the run, with the legacy defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct RunIdentity {
    pub character: String,
    pub seed: String,
    pub ascension: u8,
    pub backend: String,
    pub act1: String,
    /// `None` when the save predates the field (the loader must probe).
    pub run_mode: Option<bool>,
    /// The game build the line was recorded in; v0.107.1 when the save
    /// carries no `game_version` (GH #801). An unknown value also reads as
    /// the default here — `Workbench::restore_save` rejects it first.
    pub game_version: GameVersion,
    /// The run was built as the e2e student's training run (run-mode only;
    /// absent = false).
    pub e2e_setup: bool,
}

impl RunIdentity {
    #[must_use]
    pub fn from_save(payload: &Value) -> Self {
        let s = |k: &str, default: &str| {
            payload
                .get(k)
                .and_then(Value::as_str)
                .unwrap_or(default)
                .to_string()
        };
        Self {
            character: s("character", ""),
            seed: s("seed", ""),
            ascension: u8::try_from(
                payload
                    .get("ascension")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            )
            .unwrap_or(u8::MAX),
            backend: s("backend", DEFAULT_BACKEND),
            act1: s("act1", DEFAULT_ACT1),
            run_mode: payload.get("run_mode").and_then(Value::as_bool),
            game_version: crate::session::game_version_of(payload).unwrap_or_default(),
            e2e_setup: payload
                .get("e2e_setup")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }
    }

    /// The `run_mode` values to try, best first (`restore_attempts`).
    #[must_use]
    pub fn restore_attempts(&self) -> Vec<bool> {
        if let Some(rm) = self.run_mode {
            return vec![rm];
        }
        let likely = !self.seed.is_empty() && self.seed.chars().all(|c| c.is_ascii_digit());
        vec![likely, !likely]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_matches_python() {
        assert_eq!(slugify("My Save!"), "my-save");
        assert_eq!(slugify("  --A_b  "), "a-b");
        assert_eq!(slugify("***"), "save");
        assert_eq!(slugify("Act 2 boss"), "act-2-boss");
    }

    #[test]
    fn header_strips_heavy_fields() {
        let p = json!({"schema_version": 1, "name": "x", "line": [1, 2], "node_values": [null]});
        assert_eq!(header(&p), json!({"schema_version": 1, "name": "x"}));
        assert!(is_valid(&p));
        assert!(!is_valid(&json!({"schema_version": 2})));
    }

    #[test]
    fn identity_defaults_and_probe_order() {
        let ident =
            RunIdentity::from_save(&json!({"character": "Silent", "seed": "123", "ascension": 3}));
        assert_eq!(ident.backend, "csharp");
        assert_eq!(ident.act1, "overgrowth");
        assert_eq!(ident.restore_attempts(), vec![true, false]);
        let ident = RunIdentity::from_save(
            &json!({"character": "Silent", "seed": "ABC", "ascension": 0, "run_mode": false}),
        );
        assert_eq!(ident.restore_attempts(), vec![false]);
        assert_eq!(ident.game_version, GameVersion::V0_107_1);
        let ident = RunIdentity::from_save(&json!({"seed": "ABC", "game_version": "v0.111.0"}));
        assert_eq!(ident.game_version, GameVersion::V0_111_0);
    }

    #[test]
    fn build_save_names_only_a_non_default_build() {
        let progress = Progress {
            act: json!(1),
            floor: json!(0),
            hp: json!(80),
            max_hp: json!(80),
        };
        let args = |v| {
            build_save(
                "n",
                "t",
                "Ironclad",
                "ABC",
                0,
                "rust",
                false,
                "roll",
                v,
                &progress,
                &[],
                &[],
            )
        };
        let old = args(GameVersion::V0_107_1);
        assert!(old.get("game_version").is_none(), "{old}");
        let new = args(GameVersion::V0_111_0);
        assert_eq!(new["game_version"], "v0.111.0");
        assert_eq!(
            RunIdentity::from_save(&new).game_version,
            GameVersion::V0_111_0
        );
    }
}
