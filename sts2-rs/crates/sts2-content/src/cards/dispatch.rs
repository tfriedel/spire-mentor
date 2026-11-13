//! `PlayCard` dispatch registry — bead `sts2-rs-ov4` (P6-1).
//!
//! The function-pointer **registry storage** lives in
//! [`sts2_core::card_effects`] so the engine (`combat_loop::play_card`) can
//! look entries up without taking a dependency on this crate. The
//! Strike/Defend/Bash/AscendersBane handler bodies also live in core for
//! the same reason (sts2-core integration tests cannot dev-dep on
//! sts2-content without a workspace cycle).
//!
//! This file is the **extension point** for the downstream P6 character
//! pool beads (P6-3 .. P6-7 — Ironclad, Silent, Defect,
//! Colorless/Curses/Status). Each bead adds a `pub mod <character>` here
//! and registers its pool via [`sts2_core::card_effects::register_many`]
//! inside an `install_<character>_pool()` function called from
//! [`install_default_registry`].
//!
//! Source-of-truth: `sts2-rl-agent@01c094d:sts2_env/cards/*` factories.

use sts2_core::card_effects;

use super::spec_index::{
    COLORLESS_SPECS, DEFECT_SPECS, IRONCLAD_SPECS, NECROBINDER_SPECS, REGENT_SPECS, SILENT_SPECS,
    STATUS_SPECS,
};
use super::types::CardKeywords;

/// Install every card handler in the content library. Idempotent.
///
/// Today this is just the core/starter handlers (Strike / Defend / Bash /
/// `AscendersBane`), which live in [`sts2_core::card_effects`] so the core
/// crate's own test suite can install them without a workspace cycle.
/// Downstream beads (P6-3 .. P6-7) add their per-character installer
/// calls here.
pub fn install_default_registry() {
    card_effects::install_default_registry();
    // Card tails awaited behind a suspended draw (Big Bang, …).
    sts2_core::register_draw_then_runner(super::draw_then::run);
    // Bead `sts2-rs-63mew`: C#-discriminant rarity ranks for core-side
    // sorts (Stratagem's `after_shuffle` draw-pile grid-select orders
    // its options `orderby c.Rarity, c.Id`).
    sts2_core::register_card_rarity_rank(super::spec_index::card_rarity_rank_csharp_in);
    // Bead `sts2-rs-RP22`: C# `IsRemovable => !Eternal` filter for the generic
    // deck-removal offer (`CardSelectCmd.FromDeckForRemoval`), so Eternal
    // curses (Greed/AscendersBane) don't occupy a removal-offer index.
    sts2_core::register_card_removable(super::spec_index::card_is_removable_instance);
    // GH #793: the game-build-aware card factory behind `Card::upgrade_for`
    // and `Card::normalize_for_version` (v0.107.1 never reaches it).
    sts2_core::register_card_factory_for_version(super::v0111::card_pod_for);
    super::ironclad::install_pool();
    super::silent::install_pool();
    super::defect::install_pool();
    super::colorless::install_pool();
    super::status::install_pool();
    super::curse::install_pool();
    super::necrobinder::install_pool(); // rebuilt from scratch (C#-first)
    super::regent::install_pool();
    super::fool::install_pool();

    // Bead `sts2-rs-6ynt.3.4`: populate the engine-side EXHAUST keyword
    // registry so `combat_loop::play_card` routes EXHAUST-keyworded cards
    // to the Exhaust pile instead of Discard. Iterates every per-character
    // spec slice (Regent added via the Regent parity coverage pack — its 88
    // cards carry EXHAUST/ETHEREAL/INNATE keywords that were silently dropped
    // while REGENT_SPECS was missing from these loops).
    // Bead `sts2-rs-ut1v` Part A: install the character-pool card sampler
    // trampoline so sts2-core hooks (CreativeAi.after_side_turn_start —
    // future InfiniteBlades / Outbreak / Predator) can spawn random
    // character cards into hand via Effect::SpawnCharacterCardIntoHand.
    card_effects::register_character_card_sampler(
        super::colorless::sample_character_cards_for_hook,
    );
    // Bead `sts2-rs-q6bix`: with-replacement sibling (C# `GetForCombat`)
    // for `CalamityPower.after_card_played`, which conjures `Amount`
    // random Attacks (uniform per-slot pick) each time the owner plays an
    // Attack — distinct from the distinct-shuffle trampoline above.
    card_effects::register_character_card_replacement_sampler(
        super::colorless::sample_character_cards_with_replacement_for_hook,
    );
    // GH #636: Common-rarity sibling trampoline for
    // `HelloWorldPower.BeforeHandDraw`, whose C# LINQ narrows the owner's
    // character pool with `where c.Rarity == CardRarity.Common` before
    // `GetDistinctForCombat` shuffles it. `CardRarity` is content-local, so
    // the rarity predicate cannot ride the `card_type` parameter of the
    // sampler above and gets its own OnceLock.
    card_effects::register_character_common_card_sampler(
        super::colorless::sample_character_common_cards_for_hook,
    );
    // Bead `sts2-rs-fa2o`: ethereal-only sibling trampoline so the
    // BigHat `after_side_turn_start` body in core can sample ethereal
    // cards from the player's character pool. The two trampolines are
    // independent OnceLocks; `install_ethereal_keywords` below
    // populates `card_effects::card_ethereal(id)` which the
    // ethereal sampler filters on.
    card_effects::register_character_ethereal_card_sampler(
        super::colorless::sample_character_ethereal_cards_for_hook,
    );
    // Bead `sts2-rs-7xih`: colorless-pool sampler. Character-
    // independent (single trampoline, no `CharacterId` param) — feeds
    // OrangeDough's `after_side_turn_start` body and any future
    // colorless-spawn hooks (VexingPuzzlebox — bead `sts2-rs-t7vn`).
    // Samples WITH replacement from `COLORLESS_SPECS` to match Python's
    // `generate_colorless_cards`.
    card_effects::register_colorless_card_sampler(
        super::colorless::sample_colorless_cards_for_hook,
    );
    // Distinct colorless sampler (C# `GetDistinctForCombat` over the filtered
    // ColorlessCardPool via the CombatCardGeneration rng) — feeds
    // SpectrumShiftPower's `BeforeHandDraw` per-turn colorless generation.
    card_effects::register_distinct_colorless_card_sampler(
        super::colorless::sample_distinct_colorless_cards_for_hook,
    );

    // Bead `sts2-rs-isooo` (q6bix ENTROPY): transform-to-random trampoline.
    // `combat_loop::start_player_turn` calls
    // `card_effects::transform_card_to_random` from the EntropyPower
    // turn-start block; the impl lives here so the transform-options pool
    // builder (`combat_generatable_pool`) stays content-side.
    card_effects::register_transform_to_random(
        super::colorless::helpers::transform_card_to_random_for_hook,
    );

    // Bead `sts2-rs-jxz`: Regent forge trampoline. `FurnacePower`'s
    // `after_side_turn_start` body in core defers a forge through
    // `Effect::Forge`, resolved via `card_effects::forge` → this content-side
    // `combat_forge` (which must mint a `SovereignBlade` from a content
    // `CardSpec`). Card-triggered forges (TheSmith / RefineBlade / …) call
    // `combat_forge` directly; this only wires the power-triggered path.
    card_effects::register_forge(super::regent::helpers::combat_forge);

    install_exhaust_keywords();
    // Bead `sts2-rs-6ynt.3.3`: sibling pass for the ETHEREAL keyword.
    // `combat_loop::end_player_turn_phase_two` consults this registry
    // when expelling ethereal hand cards to the Exhaust pile.
    install_ethereal_keywords();
    // Bead `sts2-rs-6ynt.8-unplayable`: third sibling pass. Card bodies
    // that filter pile contents by playability (Catastrophe) consult
    // `card_effects::card_unplayable(id)`.
    install_unplayable_keywords();
    // Bead `sts2-rs-2un`: fourth sibling pass for the INNATE keyword.
    // `combat_loop::start_player_turn` consults `card_effects::card_innate`
    // to sort Innate cards to the top of the round-1 opening draw.
    install_innate_keywords();
    // Bead `sts2-rs-o8ixl`: fifth sibling pass for the RETAIN keyword.
    // `combat_loop::end_player_turn_phase_two` consults
    // `card_effects::card_retain_inst` to keep Retain hand cards
    // (Restlessness, …) IN hand at end of turn instead of flushing them.
    install_retain_keywords();
    // SLY keyword pass: a Sly card auto-plays when discarded by
    // CalculatedGamble (`CardCmd.DiscardAndDraw`). `play_calculated_gamble`
    // consults `card_effects::card_sly` to collect-then-auto-play.
    install_sly_keywords();

    // Bead `sts2-rs-h9u` (Phase-7 sliver): enemy on-death "second phase"
    // spawns. `InfestedPower.AfterDeath` summons 4 stunned Wrigglers when
    // the PhrogParasite elite dies; the handler body lives in
    // `monsters::act1_overgrowth` (so the wrigglers get their real AI) and
    // is fired by `combat_loop::check_combat_end`'s all-enemies-dead branch.
    sts2_core::enemy_death_effects::register(
        sts2_core::powers::PowerKind::Infested,
        crate::monsters::act1_overgrowth::infested_after_death_spawn,
    );

    // Bead `sts2-rs-hjpnj`: `AdaptablePower.AfterDeath` revives the
    // TestSubject boss up to twice (SecondFormHp = 200, ThirdFormHp = 300).
    // C# `AdaptablePower.cs:32-39` + `TestSubject.cs:252-278`. Unlike
    // InfestedPower (one-shot spawn), Adaptable needs to fire on the first
    // AND second deaths; `fire_pending` skips the post-handler amount-zero
    // when the owner is revived (current_hp > 0 after handler returns).
    sts2_core::enemy_death_effects::register(
        sts2_core::powers::PowerKind::Adaptable,
        crate::monsters::act3_glory::adaptable_after_death,
    );

    // Bead `sts2-rs-zo6fk`: `StockPower.AfterDeath` respawns a fresh Axebot
    // at the dead bot's slot the instant it dies (C# `StockPower.cs`). Unlike
    // Infested/Adaptable (which fire from the combat-end `fire_pending`
    // sweep), Stock must fire mid-combat while a sibling Axebot is still
    // alive — so it is fired per-death via `fire_for_owner` from the
    // `just_died` chokepoint in `damage.rs`. The respawn decrements the
    // stock (2 → 1 → 0), capping at three bots per slot.
    sts2_core::enemy_death_effects::register(
        sts2_core::powers::PowerKind::Stock,
        crate::monsters::act3_glory::axebot_stock_after_death,
    );

    // Bead `sts2-rs-jwb4a`: `PossessStrengthPower`/`PossessSpeedPower.AfterDeath`
    // give the player back the Strength/Dexterity TheLost/TheForgotten stole
    // (DEBILITATING_SMOG / MIASMA) when the thief dies. C#
    // `PossessStrengthPower.cs` accumulates every negative stat applied to the
    // player and reverts it on `AfterDeath`. Restoring no creature, these never
    // keep combat open (`fire_pending` only does so when an enemy is alive
    // after firing). Fixes the THE_LOST_AND_FORGOTTEN `player.powers` /
    // `player_state` fuzz divergences where Rust left the stolen debuff after
    // the thief died (e.g. v2 seed 2042: HOWL_FROM_BEYOND kills TheLost, C#
    // drops the player's -2 Strength, Rust kept it).
    sts2_core::enemy_death_effects::register(
        sts2_core::powers::PowerKind::PossessStrength,
        crate::monsters::act3_glory::possess_strength_after_death,
    );
    sts2_core::enemy_death_effects::register(
        sts2_core::powers::PowerKind::PossessSpeed,
        crate::monsters::act3_glory::possess_speed_after_death,
    );

    // Bead `sts2-rs-surprise-death-spawn`: `SurprisePower.AfterDeath`
    // (`MegaCrit.Sts2.Core.Models.Powers/SurprisePower.cs:18-43`) spawns a
    // SneakyGremlin + FatGremlin when the GremlinMerc dies, and its
    // `ShouldStopCombatFromEnding() => true` keeps the fight open.
    // `GremlinMercNormal.GenerateMonsters` spawns ONLY the merc, so without
    // this hook the engine scored a PHANTOM VICTORY the instant the merc
    // dropped (mirror corpus records 71 / 218). Fires per-death via
    // `fire_for_owner` from the `just_died` chokepoint (C# runs `AfterDeath`
    // inside the death pipeline); `check_combat_end`'s live-enemy re-check
    // after `fire_pending` supplies the `ShouldStopCombatFromEnding` contract.
    sts2_core::enemy_death_effects::register(
        sts2_core::powers::PowerKind::Surprise,
        crate::monsters::act1_underdocks::surprise_after_death_spawn,
    );

    // Every on-play handler and late effect is registered by now: publish the
    // dense `CardId`-indexed copies `lookup` / `fire_late_effects` read on
    // every card play. This must stay the LAST step. A re-install (production
    // runs two) re-registers the same handlers, which suspends the tables
    // until this call re-checks them against the registries (M06,
    // `docs/perf/2026-10-13-engine-speed-exact-design.md`).
    card_effects::freeze_card_tables();
}

/// Walk every generated `CardSpec` slice and register the `EXHAUST` keyword
/// in the engine-side metadata registry. Idempotent.
fn install_exhaust_keywords() {
    let slices: &[&[fn() -> super::types::CardSpec]] = &[
        IRONCLAD_SPECS,
        SILENT_SPECS,
        DEFECT_SPECS,
        NECROBINDER_SPECS,
        REGENT_SPECS,
        COLORLESS_SPECS,
        STATUS_SPECS,
        // v0.111.0-only ids (GH #793): absent from every v0.107.1 pool, so
        // registering their keywords cannot touch a v0.107.1 card.
        super::spec_index::V0111_ONLY_SPECS,
    ];
    for slice in slices {
        for &factory in *slice {
            let spec = factory();
            if spec.keywords.contains(CardKeywords::EXHAUST) {
                card_effects::register_card_exhausts(spec.id);
            }
        }
    }
}

/// Walk every generated `CardSpec` slice and register the `ETHEREAL` keyword
/// in the engine-side metadata registry. Idempotent. Bead `sts2-rs-6ynt.3.3`.
///
/// The four canonical ETHEREAL cards at pin `01c094d` (Dazed, Void,
/// AscendersBane, Clumsy) all live in `STATUS_SPECS`. The curse pool
/// (`super::curse::install_pool`) only carries on-play handler fn-pointers;
/// the spec data — and thus the keyword bits — comes from
/// `status_specs_generated.rs`.
/// Walk every generated `CardSpec` slice and register the `UNPLAYABLE`
/// keyword in the engine-side metadata registry. Idempotent. Bead
/// `sts2-rs-6ynt.8-unplayable`. At pin `01c094d` the UNPLAYABLE cards
/// are the 5 status cards (Wound/Dazed/Burn/Void/Slimed-no-Slimed-is-
/// playable — Wound/Dazed/Burn/Void), AscendersBane, and every curse.
fn install_unplayable_keywords() {
    let slices: &[&[fn() -> super::types::CardSpec]] = &[
        IRONCLAD_SPECS,
        SILENT_SPECS,
        DEFECT_SPECS,
        NECROBINDER_SPECS,
        REGENT_SPECS,
        COLORLESS_SPECS,
        STATUS_SPECS,
        // v0.111.0-only ids (GH #793): absent from every v0.107.1 pool, so
        // registering their keywords cannot touch a v0.107.1 card.
        super::spec_index::V0111_ONLY_SPECS,
    ];
    for slice in slices {
        for &factory in *slice {
            let spec = factory();
            if spec.keywords.contains(CardKeywords::UNPLAYABLE) {
                card_effects::register_card_unplayable(spec.id);
            }
        }
    }
}

fn install_ethereal_keywords() {
    let slices: &[&[fn() -> super::types::CardSpec]] = &[
        IRONCLAD_SPECS,
        SILENT_SPECS,
        DEFECT_SPECS,
        NECROBINDER_SPECS,
        REGENT_SPECS,
        COLORLESS_SPECS,
        STATUS_SPECS,
        // v0.111.0-only ids (GH #793): absent from every v0.107.1 pool, so
        // registering their keywords cannot touch a v0.107.1 card.
        super::spec_index::V0111_ONLY_SPECS,
    ];
    for slice in slices {
        for &factory in *slice {
            let spec = factory();
            if spec.keywords.contains(CardKeywords::ETHEREAL) {
                card_effects::register_card_ethereal(spec.id);
            }
        }
    }
}

/// Walk every generated `CardSpec` slice and register the `INNATE`
/// keyword in the engine-side metadata registry. Idempotent. Consumed by
/// the round-1 opening-draw sort in `combat_loop::start_player_turn`
/// (C# moves Innate draw-pile cards to the top before the first draw).
/// Bead `sts2-rs-2un`. MindBlast / DramaticEntrance / Brutality etc.
fn install_innate_keywords() {
    let slices: &[&[fn() -> super::types::CardSpec]] = &[
        IRONCLAD_SPECS,
        SILENT_SPECS,
        DEFECT_SPECS,
        NECROBINDER_SPECS,
        REGENT_SPECS,
        COLORLESS_SPECS,
        STATUS_SPECS,
        // v0.111.0-only ids (GH #793): absent from every v0.107.1 pool, so
        // registering their keywords cannot touch a v0.107.1 card.
        super::spec_index::V0111_ONLY_SPECS,
    ];
    for slice in slices {
        for &factory in *slice {
            let spec = factory();
            if spec.keywords.contains(CardKeywords::INNATE) {
                card_effects::register_card_innate(spec.id);
            }
        }
    }
}

/// Walk every generated `CardSpec` slice and register the `RETAIN`
/// keyword in the engine-side metadata registry. Idempotent. Consumed by
/// the end-of-turn flush in `combat_loop::end_player_turn_phase_two`
/// (C# partitions the hand by `CardModel.ShouldRetainThisTurn`, which is
/// true for any card whose `Keywords.Contains(Retain)`, keeping those
/// cards IN hand while the rest flush to discard). Bead `sts2-rs-o8ixl`
/// (q6bix RESTLESSNESS). Restlessness is the canonical base-keyword Retain
/// card; others gain RETAIN only on upgrade and are handled per-instance
/// via `Card::keyword_override` (`card_retain_inst`).
fn install_retain_keywords() {
    let slices: &[&[fn() -> super::types::CardSpec]] = &[
        IRONCLAD_SPECS,
        SILENT_SPECS,
        DEFECT_SPECS,
        NECROBINDER_SPECS,
        REGENT_SPECS,
        COLORLESS_SPECS,
        STATUS_SPECS,
        // v0.111.0-only ids (GH #793): absent from every v0.107.1 pool, so
        // registering their keywords cannot touch a v0.107.1 card.
        super::spec_index::V0111_ONLY_SPECS,
    ];
    for slice in slices {
        for &factory in *slice {
            let spec = factory();
            if spec.keywords.contains(CardKeywords::RETAIN) {
                card_effects::register_card_retain(spec.id);
            }
        }
    }
}

/// Register every spec carrying the `SLY` keyword (C# `CardKeyword.Sly`).
/// `play_calculated_gamble` reads `card_effects::card_sly` to replicate
/// `CardCmd.DiscardAndDraw`'s collect-then-auto-play of discarded Sly cards.
/// All Sly cards are Silent, so the registry is inert for other characters.
fn install_sly_keywords() {
    let slices: &[&[fn() -> super::types::CardSpec]] = &[
        IRONCLAD_SPECS,
        SILENT_SPECS,
        DEFECT_SPECS,
        NECROBINDER_SPECS,
        REGENT_SPECS,
        COLORLESS_SPECS,
        STATUS_SPECS,
        // v0.111.0-only ids (GH #793): absent from every v0.107.1 pool, so
        // registering their keywords cannot touch a v0.107.1 card.
        super::spec_index::V0111_ONLY_SPECS,
    ];
    for slice in slices {
        for &factory in *slice {
            let spec = factory();
            if spec.keywords.contains(CardKeywords::SLY) {
                card_effects::register_card_sly(spec.id);
            }
        }
    }
}

/// Idempotent alias for [`install_default_registry`]. Convenient for tests
/// that need the registry populated but don't care which beads contributed.
pub fn ensure_installed() {
    install_default_registry();
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sts2_core::card_effects::{dispatch, lookup};
    use sts2_core::cards::CardId;
    use sts2_core::{CombatState, EffectResult};

    /// Strike/Defend/Bash/AscendersBane are all registered (NOT
    /// `NotImplemented`) once `install_default_registry` has run.
    #[test]
    fn strike_defend_bash_in_registry() {
        ensure_installed();
        assert!(lookup(CardId::StrikeIronclad).is_some());
        assert!(lookup(CardId::DefendIronclad).is_some());
        assert!(lookup(CardId::Bash).is_some());
        assert!(lookup(CardId::AscendersBane).is_some());
    }

    /// Bead `sts2-rs-o8ixl` (q6bix RESTLESSNESS): `install_default_registry`
    /// populates the RETAIN keyword registry from the per-character spec
    /// slices. After install, Restlessness (the canonical base-keyword
    /// Retain card) reports `card_retain == true`, while a plain Strike
    /// does not — so the end-of-turn flush keeps Restlessness in hand.
    #[test]
    fn restlessness_registered_as_retain() {
        ensure_installed();
        assert!(
            card_effects::card_retain(CardId::Restlessness),
            "Restlessness must be registered as a Retain card"
        );
        assert!(
            !card_effects::card_retain(CardId::StrikeIronclad),
            "a plain Strike must not be a Retain card"
        );
    }

    /// Kills `install_innate_keywords -> ()`: after install, DramaticEntrance
    /// (a canonical base-keyword INNATE card) reports `card_innate == true`,
    /// while a plain Strike does not. If the installer no-ops, the registry
    /// stays empty and DramaticEntrance is not flagged innate.
    #[test]
    fn dramatic_entrance_registered_as_innate() {
        ensure_installed();
        assert!(
            card_effects::card_innate(CardId::DramaticEntrance),
            "DramaticEntrance must be registered as an Innate card"
        );
        assert!(
            !card_effects::card_innate(CardId::StrikeIronclad),
            "a plain Strike must not be an Innate card"
        );
    }

    /// Kills `install_unplayable_keywords -> ()`: after install, AscendersBane
    /// reports `card_unplayable == true`, while a plain Strike does not.
    #[test]
    fn ascenders_bane_registered_as_unplayable() {
        ensure_installed();
        assert!(
            card_effects::card_unplayable(CardId::AscendersBane),
            "AscendersBane must be registered as Unplayable"
        );
        assert!(
            !card_effects::card_unplayable(CardId::StrikeIronclad),
            "a plain Strike must not be Unplayable"
        );
    }

    /// Kills `install_sly_keywords -> ()`: after install, SneakyCard (a Silent
    /// SLY card) reports `card_sly == true`, while a plain Strike does not.
    #[test]
    fn sneaky_registered_as_sly() {
        ensure_installed();
        assert!(
            card_effects::card_sly(CardId::SneakyCard),
            "SneakyCard must be registered as a Sly card"
        );
        assert!(
            !card_effects::card_sly(CardId::StrikeIronclad),
            "a plain Strike must not be a Sly card"
        );
    }

    /// Bead `sts2-rs-h9u`: `install_default_registry` also wires the
    /// enemy on-death "second phase" handlers. After install, the
    /// `Infested` power (PhrogParasite elite) resolves to a handler so the
    /// 4-Wriggler second phase fires in a live run.
    #[test]
    fn infested_on_death_handler_registered() {
        ensure_installed();
        assert!(
            sts2_core::enemy_death_effects::lookup(sts2_core::powers::PowerKind::Infested)
                .is_some(),
            "InfestedPower.AfterDeath handler must be installed by default"
        );
    }

    /// A representative un-ported card returns `NotImplemented`.
    #[test]
    fn unimplemented_card_returns_not_implemented() {
        ensure_installed();
        // `CardId::Generic` is the sentinel/placeholder variant — it has no
        // play-handler in any pool (and never will), so `lookup` returns None
        // and dispatch yields NotImplemented. It is the only unregistered
        // CardId once every character pool is installed, making it a stable
        // permanent anchor. (Earlier examples — BansheesCry → Eradicate → Wisp
        // — were retired as the Necrobinder pool finished landing.)
        let mut state = CombatState::default();
        assert_eq!(
            dispatch(&mut state, CardId::Generic, None),
            EffectResult::NotImplemented
        );
    }

    /// Every `CardId` variant either dispatches to a real handler or returns
    /// `NotImplemented` on lookup — never panics. Calling un-registered
    /// `Enemy`-target handlers on an empty state would panic (no Play-pile
    /// entry), so this test verifies the lookup contract only; the engine's
    /// `play_card` integration tests in `crates/sts2-core/tests/play_card.rs`
    /// cover the call path for the shipped cards.
    #[test]
    fn all_variants_dispatch_or_not_implemented() {
        ensure_installed();
        for &id in CardId::all() {
            let _ = lookup(id);
            let _ = id.to_python_name();
        }
    }

    /// Exhaustive: every variant has a non-empty upper-snake-case
    /// `to_python_name`. Adding a `CardId` variant without extending the
    /// generated `to_python_name` arm is a compile error (the generated
    /// `match` is exhaustive); this test catches silent regenerations
    /// that drop variants and pins the format.
    #[test]
    fn every_variant_has_python_name() {
        for &id in CardId::all() {
            let name = id.to_python_name();
            assert!(!name.is_empty(), "empty python_name for {id:?}");
            assert!(
                name.bytes()
                    .all(|b| b == b'_' || b.is_ascii_uppercase() || b.is_ascii_digit()),
                "non upper-snake-case python_name for {id:?}: {name:?}"
            );
        }
        assert_eq!(CardId::StrikeIronclad.to_python_name(), "STRIKE_IRONCLAD");
        assert_eq!(CardId::Bash.to_python_name(), "BASH");
        assert_eq!(CardId::AscendersBane.to_python_name(), "ASCENDERS_BANE");
    }

    /// `CardId` variant count is what we think it is (sanity guard).
    #[test]
    fn card_id_variant_count_is_stable() {
        // Python `sts2_env/core/enums.py::CardId` had 579 members at the
        // (now archived) pin `01c094d`; Grapple was removed (not in C#) → 578.
        // Plus 1 C#-only addition (NotYet, absent from the Python pin) → 579.
        // v0.107.1 content port: +1 `Wither` status card (FollowThrough→Scare
        // is a rename, no net change) → 580.
        // `card_ids_generated.rs` is hand-maintained against the v0.107.1 C#
        // decompilation — bump this count in the same change that adds or
        // removes a variant. This assertion is the only guard.
        assert_eq!(CardId::all().len(), 580);
    }
}
