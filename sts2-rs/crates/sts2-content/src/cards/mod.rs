// HAND-MAINTAINED — edit directly. Add a `pub mod` line when you add a pool.
//! Card content: static data definitions for every card in the pool.
//!
//! Shared types (`CardSpec`, `CardEffect`, `CardKeywords`, `CardRarity`)
//! live in `types.rs` and are re-exported here for backwards-compat with
//! every `use crate::cards::CardSpec` already on disk.
//!
//! Bead sts2-rs-pmf (P6-1b) originally generated this file from `cards.toml`
//! via a `scripts/regen-cards-mod.py` that no longer exists. It was retired
//! once the file outgrew what the manifest could express: `spec_index` and
//! `card_upgrades_generated` are hand-added, the per-class
//! `<char>_specs_generated` modules the script still emitted were deleted
//! long ago, and two pool comments carry C#-port history. A rerun therefore
//! declared four missing modules and dropped three real ones — it could not
//! produce a file that compiled. `cards.toml` survives as a provenance
//! table only; nothing reads it.

pub mod dispatch;
pub mod draw_then;
// What a card face shows for a value the card only knows at play time.
pub mod preview;
pub mod types;

pub use types::{CardEffect, CardKeywords, CardRarity, CardSpec};

// P4-1 (sts2-rs-zcl)
pub mod ironclad_basic;

// P6-3 (sts2-rs-le3)
//   installer: ironclad::install_pool
pub mod ironclad;

// P6-4 (sts2-rs-l2c)
//   installer: silent::install_pool
pub mod silent;

// P6-5 (sts2-rs-l7s)
//   installer: defect::install_pool
pub mod defect;

// P6-7 (sts2-rs-ltq)
//   installer: colorless::install_pool
pub mod colorless;

// P6-7 (sts2-rs-ltq)
//   installer: status::install_pool
pub mod status;

// P6-7 (sts2-rs-ltq)
//   installer: curse::install_pool
pub mod curse;

// Necrobinder card pool — rebuilt from scratch (C#-first) via the
// character-port skill after the prior Python-pin-derived pool was deleted.
pub mod necrobinder;

// bay-regent (sts2-rs-bay; plumbing bead sts2-rs-qeo)
//   installer: regent::install_pool
pub mod regent;

// v0.113.0 The Fool.
//   installer: fool::install_pool
pub mod fool;

// Generated POD spec modules — bead sts2-rs-pmf (P6-1b).
// Compiled to catch CardSpec-shape regressions; not re-exported.
pub(crate) mod card_upgrades_generated;
pub(crate) mod status_specs_generated;

// Aggregate slices over the per-class generated factories above.
// Hand-rolled (357 entries total) so resolve_card_name and the
// `--deck` parser can see every card whose dispatch handler exists.
// Bead `sts2-rs-pjln` (cliext H4).
pub mod spec_index;

// v0.111.0 card values / rarities / keywords / pools overlay (GH #793).
pub mod v0111;

// The C# pools in `GenerateAllCards()` order and the co-op generation pools
// derived from them (GH #991).
pub mod pool_order;

/// The Fool's card pool seam (v0.113.0); see the module doc.
pub mod fool_pool;
