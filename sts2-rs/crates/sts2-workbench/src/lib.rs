//! The browser workbench's engine-facing layer.
//!
//! Port of `sts2-pycli/python/workbench` (the FastAPI layer over the PyO3
//! engine) into plain Rust: every entry point takes and returns
//! `serde_json::Value` shaped exactly like today's HTTP routes, so the
//! React SPA keeps its request/response contracts whether it talks to the
//! Python server (Cloud Run) or to this crate compiled to wasm inside a
//! Web Worker.
//!
//! Deliberately NO filesystem, threads, clocks, or platform bindings in
//! this crate — those live in `sts2-workbench-wasm` (browser) and, later,
//! a native coach binary. That is what lets the golden-parity tests run
//! under `cargo nextest` natively.

pub mod combatlog;
pub mod options;
pub mod playlog;
pub mod ppo;
pub mod render;
pub mod replay;
pub mod seed;
pub mod statediff;

pub mod advisors;
pub mod api;
pub mod bench;
pub mod combat_import;
pub mod combat_view;
pub mod deck;
#[cfg(feature = "e2e")]
pub mod e2e;
pub mod enemy_labels;
pub mod enrich;
pub mod game;
pub mod rmx;
pub mod runlog;
pub mod runmode;
pub mod saves;
pub mod session;
pub mod session_choice;
#[cfg(feature = "e2e")]
mod session_e2e;
mod session_runmode;
pub mod simlab;
pub mod simlab_catalog;
pub mod simlab_sweep;
pub mod state_import;
pub use session_runmode::Clock;
pub mod step;

pub mod assets;
pub mod data;
pub mod display_maps;
pub mod localization;
