//! Relic listener-installation entry point — bead `sts2-rs-l7l` (P7-1).
//!
//! Mirrors `sts2-content::cards::dispatch` (bead `sts2-rs-ov4` / P6-1).
//! The typed hook-channel **registry storage** lives in
//! [`sts2_core::relic_effects`] (and the run-side channels in
//! `sts2_core::run_effects`) so the in-combat / run-loop dispatchers
//! (`relic_effects::dispatch_after_combat_victory`, …) can look entries
//! up without taking a dependency on this crate. The three starter
//! handlers (`CrackedCore`, `RingOfTheSnake`, `DivineRight`) also live
//! in core for the same reason (sts2-core integration tests cannot
//! dev-dep on sts2-content without a workspace cycle).
//!
//! Each pool module exposes an `install_pool()` that registers its
//! per-relic listeners onto the relevant typed channels; this file
//! calls them all from [`install_default_registry`]. (The vestigial
//! generic `on_trigger` marker layer was removed in bead
//! `sts2-rs-j87a7` — relics are bundles of typed hook listeners, with
//! no "trigger relic by id" entry point.)
//!
//! Source-of-truth: the C# decomp at `../sts2-wiki/decompiled/v0.103.2`;
//! the per-pool ports also pin `sts2-rl-agent@01c094d:sts2_env/relics/*`.

use sts2_core::relic_effects;

/// Install every relic handler in the content library, then freeze the relic
/// hook channels. Idempotent (guarded by its own `Once`).
///
/// The core handlers (the starters `CrackedCore`, `RingOfTheSnake`,
/// `DivineRight` and the rest of [`sts2_core::relic_effects`]) live in core so
/// the core crate's own test suite can install them without a workspace
/// cycle; the per-pool installers below register the content relics on top.
///
/// The `Once` is load-bearing for speed, not just tidiness: production calls
/// this twice (the PyO3 module init and `sts2_runai::rollout::
/// ensure_content_installed`), and a second run would re-register every
/// handler AFTER the freeze, un-freezing those channels for the rest of the
/// process (M02).
pub fn install_default_registry() {
    static INSTALL_ONCE: std::sync::Once = std::sync::Once::new();
    INSTALL_ONCE.call_once(|| {
        relic_effects::install_default_registry();
        // Register the card-pod factory so `sts2-core`'s
        // `ChoiceKind::AddGeneratedToHand` resolver can build cards from
        // `(CardId, upgraded)` without a workspace cycle. Used by Toolbox /
        // ChoicesParadox listeners that store synthetic options on
        // `PendingChoice`.
        sts2_core::register_card_factory(crate::cards::spec_index::card_pod_from_id);
        sts2_core::register_card_factory_for_version(crate::cards::v0111::card_pod_for);
        crate::relics::starter::install_pool();
        crate::relics::common::install_pool();
        // P7-3 (sts2-rs-l22): registered below.
        crate::relics::uncommon::install_pool();
        // P7-4 (sts2-rs-ge1): registered below.
        crate::relics::rare::install_pool();
        // P7-5 (sts2-rs-yyw): registered below.
        crate::relics::shop_event::install_pool();
        // P7-5b (sts2-rs-7ez): registered below.
        crate::relics::event_reward::install_pool();
        // W9 (sts2-rs-4rwv4): Neow-pool Ancient relics not in the Python sim.
        crate::relics::neow_relics::install_pool();
        // v0.113.0 Fool relics.
        crate::relics::fool::install_pool();
        // Every relic hook channel is now fully populated and never mutated
        // again in production: publish each channel's immutable `&'static`
        // read view so dispatch is a cheap `OnceLock::get` instead of an
        // `ArcSwap::load` (bead sts2-rs-05mjm). This must be the LAST
        // registration step — the core installer used to freeze before the
        // pools above ran, which left 25 of the 53 channels (most of the hot
        // ones) on the `ArcSwap` fallback (M02,
        // `docs/perf/2026-10-13-engine-speed-exact-design.md`).
        relic_effects::freeze_all_channels();
    });
}

/// Idempotent alias for [`install_default_registry`]. Convenient for
/// tests that need the registry populated but don't care which beads
/// contributed.
pub fn ensure_installed() {
    install_default_registry();
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sts2_core::relics::RelicId;

    /// Starter relics resolve through their typed hook channels once
    /// `install_default_registry` has run: `CrackedCore` /
    /// `RingOfTheSnake` / `DivineRight` each register a real combat-hook
    /// listener (lightning channel / +2 round-1 draw / per-turn energy).
    /// Crate-wide listener coverage is asserted by the completeness
    /// census in `tests/relic_listener_completeness.rs`.
    #[test]
    fn starter_relics_register_typed_listeners() {
        ensure_installed();
        assert!(relic_effects::lookup_before_side_turn_start(RelicId::CrackedCore).is_some());
        assert!(relic_effects::lookup_modify_hand_draw(RelicId::RingOfTheSnake).is_some());
        assert!(relic_effects::lookup_after_side_turn_start(RelicId::DivineRight).is_some());
    }

    /// Façade installer is idempotent across multiple boot paths — a
    /// re-install must not panic or double-register (the typed channels
    /// overwrite the same fn-ptr on re-register).
    #[test]
    #[serial_test::serial(relic_registry)]
    fn install_default_registry_is_idempotent() {
        ensure_installed();
        ensure_installed();
        ensure_installed();
        // No panic + the listener still resolves after repeated installs.
        assert!(relic_effects::lookup_before_side_turn_start(RelicId::CrackedCore).is_some());
    }

    /// After the content install EVERY relic hook channel serves its frozen
    /// `&'static` read view, and a second install (production runs two: the
    /// PyO3 module init and `ensure_content_installed`) leaves it that way.
    /// Fails on the pre-M02 ordering (the core installer froze, then the
    /// content pools registered on top and un-froze 25 channels) and if
    /// the install-once guard is dropped (the re-registration un-freezes).
    #[test]
    fn every_channel_frozen_after_content_install() {
        ensure_installed();
        assert!(relic_effects::all_channels_frozen());
        ensure_installed();
        assert!(relic_effects::all_channels_frozen());
    }
}
