//! The e2e Ironclad student without Python (plan
//! `docs/plans/2026-11-06-workbench-e2e-brain.md`).
//!
//! - [`net`]: the policy forward of `ppo.e2e.net.E2ENet` over an `.e2ew`
//!   weight file (`ppo.e2e.export_wb weights`).
//! - [`env`]: `ppo.e2e.env.E2EEnv`'s decision driver (auto-advance, merged
//!   classes, selections, the per-fight intent log).
//! - [`tokens`]: `ppo.e2e.tokens.encode` (policy fields only).
//!
//! Parity fixtures come from `ppo.e2e.export_wb fixtures` (Python plays the
//! seed; every decision's env view, policy row and logits are recorded).

/// Content hash of this crate's source and baked-in tables, stamped by `build.rs`
/// (`sts2_srchash::ENGINE_CRATES`); see `sts2_core::SRC_HASH`.
pub const SRC_HASH: &str = env!("STS2_E2E_SRC_HASH");

pub mod contracts;
pub mod env;
pub mod intent_log;
pub mod net;
pub mod pyjson;
pub mod schema;
pub mod sctx;
pub mod snapshot;
pub mod tactical;
pub mod tokens;
pub mod train;
pub mod vocab;
