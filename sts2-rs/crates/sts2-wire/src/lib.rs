//! `sts2-wire` — the wire dialect, parsed without a Python interpreter.
//!
//! The hint/bridge wire format (a JSON dict describing one combat decision
//! point) used to be parsed by code that lived inside the `PyO3` binding
//! crate and read `PyDict` directly. That worked, but it pinned the whole
//! reconstruction path — parse *and* `CombatState` build, ~4k lines — to
//! having `libpython` in the process. Anything else that wanted to ingest a
//! captured wire dict (a Rust test, a fuzzer, a fixture replayer, a future
//! non-Python advisor) had to boot an interpreter to do it.
//!
//! This crate is that code with the Python removed. The parse reads through
//! the [`WireValue`] trait; `serde_json::Value` implements it here and
//! `sts2-py` implements it for `Bound<'_, PyAny>`, so both hosts run the
//! *same* parser rather than two that drift. [`WireError`] carries the
//! fail-loud path/class that `sts2-py` maps back onto its Python exception
//! types.
//!
//! ```
//! let dict: serde_json::Value = serde_json::from_str(r#"{"wire_version": 2}"#).unwrap();
//! // Fails loud, with the offending path, and no interpreter involved.
//! let err = sts2_wire::parse_json(&dict).unwrap_err();
//! assert_eq!(err.path(), "$");
//! ```

/// Content hash of THIS crate's build inputs, stamped at build time by
/// `build.rs`. Compiled INTO the crate so a stale `sts2-wire` rlib carries its
/// old hash; one of the parts of `sts2_rs.src_hash()`. See
/// `sts2_core::SRC_HASH`.
pub const SRC_HASH: &str = env!("STS2_WIRE_SRC_HASH");

pub mod emit;
pub mod error;
pub mod fool_state;
pub mod graft;
pub mod scenario;
pub mod state_dict;
pub mod value;

pub use error::{schema_err, unsupported, WireError, WireResult};
pub use state_dict::{parse_state_dict, state_from_dict_impl, ParsedState};
pub use value::WireValue;

/// Parse a wire-format dict and build a `CombatState` from it — the two
/// halves ([`parse_state_dict`] + [`state_from_dict_impl`]) in one call.
///
/// # Errors
///
/// The [`parse_state_dict`] surface, plus [`WireError::Unsupported`] for any
/// unresolvable runtime spec lookup during state construction.
pub fn state_from_wire<V: WireValue>(d: &V) -> WireResult<sts2_core::CombatState> {
    let parsed = parse_state_dict(d)?;
    state_from_dict_impl(&parsed)
}

/// [`parse_state_dict`] against a `serde_json::Value`.
///
/// `WireValue` is implemented for `&Value` rather than `Value` (so element
/// access is a pointer copy, not a deep clone), which makes the generic call
/// spell out as `parse_state_dict(&&value)`. This is that, without the double
/// ampersand.
///
/// # Errors
///
/// The [`parse_state_dict`] surface.
pub fn parse_json(value: &serde_json::Value) -> WireResult<ParsedState> {
    parse_state_dict(&value)
}

/// [`state_from_wire`] against a `serde_json::Value`.
///
/// # Errors
///
/// The [`state_from_wire`] surface.
pub fn state_from_json(value: &serde_json::Value) -> WireResult<sts2_core::CombatState> {
    state_from_wire(&value)
}

/// Parse a wire-format dict straight from JSON text.
///
/// The convenience entry point for the interpreter-free callers this crate
/// exists for: a captured `combat_action` payload off disk or off a socket
/// goes in, a `CombatState` comes out.
///
/// # Errors
///
/// [`WireError::Schema`] at path `$` if the text is not valid JSON, then the
/// [`state_from_wire`] surface.
pub fn state_from_json_str(json: &str) -> WireResult<sts2_core::CombatState> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| schema_err("$", &format!("invalid JSON: {e}")))?;
    state_from_json(&value)
}
