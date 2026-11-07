//! Stamp a content hash of this crate's own source (and the `data/` tables it
//! bakes in with `include_str!`) into `STS2_E2E_SRC_HASH`, exposed as
//! `sts2_e2e::SRC_HASH`. Part of the `sts2_rs` content identity
//! (`sts2_srchash::ENGINE_CRATES`) since the mod's advisor links the student
//! into the extension; see sts2-core/build.rs for the rationale.

fn main() {
    sts2_srchash::emit_build_stamp("sts2-e2e", "STS2_E2E_SRC_HASH");
}
