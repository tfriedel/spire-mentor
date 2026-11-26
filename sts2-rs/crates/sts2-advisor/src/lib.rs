//! The live advisor's logic in Rust (GH #1426).
//!
//! The shipped advisor (`SimAdvisor`, the `sim-advisor` crate) links this;
//! it replaced the Python stack in `sts2-pycli/python` frozen by PyInstaller
//! (issue #1426). The Python modules keep their API and call the Rust ones
//! through `sts2_rs`, and the Python tests keep holding them to the old
//! behaviour.
//!
//! * [`mirror`]: the combat mirror's comparison of its prediction against
//!   reality (`combat_mirror.py`).
//! * [`mirror_state`]: what the mirror does to an engine state (carry the
//!   hidden half across a rebuild, inject a choice screen, park a turn-start
//!   resume), formerly methods of the Python `CombatState`.
//! * [`combat_mirror`]: the mirror itself, one engine state per fight kept in
//!   step with the game (the `CombatMirror` class, all but its search).
//! * [`display_names`]: player-facing names for game ids, and the guard that
//!   keeps ids off the HUD (`display_names.py`).

/// Content hash of this crate's source, stamped by `build.rs`. Part of the
/// `sts2_rs` content identity (`sts2_srchash::ENGINE_CRATES`); see
/// `sts2_core::SRC_HASH`.
pub const SRC_HASH: &str = env!("STS2_ADVISOR_SRC_HASH");

pub mod advise_combat;
pub mod advisor_mirror;
pub mod board;
pub mod board_targets;
pub mod card_select;
pub mod coach;
pub mod coach_grader;
pub mod combat_advice;
pub mod combat_mirror;
pub mod control;
pub mod coop;
pub mod crystal_sphere;
pub mod display_names;
pub mod graft;
pub mod hand_select;
pub mod headless;
pub mod i18n;
pub mod instinct;
pub mod instinct_advisor;
pub mod instinct_combat;
pub mod instinct_screens;
pub mod livetree;
pub mod mirror;
pub mod mirror_state;
pub mod policy;
pub mod py;
pub mod route;
pub mod screen_advice;
pub mod search;
pub mod serve;
pub mod session;
pub mod turn_plan;
pub mod workbench_log;
