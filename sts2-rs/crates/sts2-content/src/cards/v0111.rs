//! Card metadata and values of game build v0.111.0 (GH #793, S8a part 1).
//!
//! Every card's `spec()` keeps returning its **v0.107.1** truth, so the
//! default build is bit-identical. This module is the version overlay:
//!
//! * [`spec_for`] — the [`CardSpec`] of a card in a given build: rarity, energy
//!   cost, type, target, keywords, base damage / block. Every pool, sampler
//!   and card-construction site that works for a run or a combat reads specs
//!   through it with that state's `game_version`.
//! * [`upgrade_delta_for`] — the spec-level `OnUpgrade` delta in a build
//!   (Reave upgrades by +3 instead of +2, Mirage drops Exhaust instead of
//!   costing 1 less, …).
//! * [`card_pod_for`] — the card POD in a build, registered with core as the
//!   versioned card factory ([`sts2_core::register_card_factory_for_version`]),
//!   which `Card::upgrade_for` / `Card::normalize_for_version` use.
//! * [`pool_substitute`] / [`pool_spec`] / [`can_be_generated_in_combat`] —
//!   pool membership changes (Sidestep takes Scare's slot; `CanBeGeneratedInCombat => false`
//!   on AscendersBane, Eidolon, Nightmare, Transfigure).
//!
//! Values that live in a card's effect handler (power amounts, per-play
//! increases, HP loss, …) branch inside that handler on
//! `state.game_version` instead; see the S8a port-status table in
//! `docs/plans/2026-10-17-v0.111.0-port-inventory.md`.
//!
//! Each arm cites the v0.111.0 decomp
//! (`sts2-wiki/decompiled/v0.111.0/MegaCrit.Sts2.Core.Models.Cards/<File>.cs`).

use sts2_core::cards::{CardId, CardType, TargetType};
use sts2_core::GameVersion;

use super::types::{CardKeywords, CardRarity, CardSpec, CardUpgradeDelta};

/// `true` from game build v0.111.0 on.
#[inline]
#[must_use]
pub fn is_v0111(version: GameVersion) -> bool {
    version >= GameVersion::V0_111_0
}

/// The [`CardSpec`] of `spec.id` in game build `version`. Identity for
/// v0.107.1 and for every card whose metadata did not change.
#[must_use]
pub fn spec_for(spec: CardSpec, version: GameVersion) -> CardSpec {
    if !is_v0111(version) {
        return spec;
    }
    v0111_spec(spec).unwrap_or(spec)
}

/// [`spec_for`] over a spec factory — the shape every `*_SPECS` slice walk
/// has in hand.
#[inline]
#[must_use]
pub fn spec_of(factory: fn() -> CardSpec, version: GameVersion) -> CardSpec {
    spec_for(factory(), version)
}

/// The v0.111.0 spec of a card whose spec-level metadata changed, `None`
/// for an unchanged card.
#[allow(clippy::too_many_lines)]
fn v0111_spec(mut s: CardSpec) -> Option<CardSpec> {
    use CardRarity as R;
    match s.id {
        // ── Rarity ──────────────────────────────────────────────────────
        // v0.111.0: Rare -> Uncommon (decomp Accelerant.cs:21).
        CardId::Accelerant => s.rarity = R::Uncommon,
        // v0.111.0: Common -> Uncommon (decomp Bloodletting.cs:23).
        CardId::Bloodletting => s.rarity = R::Uncommon,
        // v0.111.0: Rare -> Uncommon (decomp Cruelty.cs:19).
        CardId::CrueltyCard => s.rarity = R::Uncommon,
        // v0.111.0: Uncommon -> Rare (decomp Dominate.cs:32).
        CardId::Dominate => s.rarity = R::Rare,
        // v0.111.0: Rare -> Uncommon (decomp EchoingSlash.cs:19).
        CardId::EchoingSlash => s.rarity = R::Uncommon,
        // v0.111.0: Uncommon -> Rare (decomp Flanking.cs:15). MP-only, so
        // no single-player pool sees it either way.
        CardId::Flanking => s.rarity = R::Rare,
        // v0.111.0: Rare -> Uncommon (decomp Ignition.cs:25). MP-only.
        CardId::Ignition => s.rarity = R::Uncommon,
        // v0.111.0: Rare -> Uncommon (decomp Salvo.cs:21).
        CardId::Salvo => s.rarity = R::Uncommon,
        // v0.111.0: Uncommon -> Rare (decomp Splash.cs:16).
        CardId::Splash => s.rarity = R::Rare,

        // ── Energy cost ─────────────────────────────────────────────────
        // v0.111.0: cost 1 -> 2 (decomp BeaconOfHope.cs:18). MP-only.
        CardId::BeaconOfHope => s.cost = 2,
        // v0.111.0: cost 2 -> 1; the damage base (15 -> 10) is a
        // `CalculationBaseVar` read by the handler (decomp Rend.cs:19,25).
        CardId::Rend => s.cost = 1,
        // v0.111.0: cost 3 -> 2 (decomp Whistle.cs:23).
        CardId::Whistle => s.cost = 2,
        // v0.111.0: cost 2 -> 3 (decomp ExpectAFight.cs:29). The Block
        // rework is S8a2's (behaviour row).
        CardId::ExpectAFight => s.cost = 3,
        // v0.111.0: cost 1 -> 2, Uncommon -> Rare (decomp WellLaidPlans.cs:12).
        CardId::WellLaidPlans => {
            s.cost = 2;
            s.rarity = R::Rare;
        }
        // v0.111.0: cost 3 -> 2 and loses Sly (decomp Haze.cs:19,32).
        CardId::Haze => {
            s.cost = 2;
            s.keywords = s.keywords.without(CardKeywords::SLY);
        }
        // v0.111.0: Uncommon Power cost 1 (Self) -> Rare Skill cost 3
        // (AllEnemies) (decomp Outbreak.cs:23). The new OnPlay is S8a2's.
        CardId::OutbreakCard => {
            s.cost = 3;
            s.card_type = CardType::Skill;
            s.target_type = TargetType::AllEnemies;
            s.rarity = R::Rare;
        }

        // ── Damage ──────────────────────────────────────────────────────
        // v0.111.0: DamageVar 11 -> 10 (decomp CollisionCourse.cs:17).
        CardId::CollisionCourse => s.base_damage = 10,
        // v0.111.0: DamageVar 7 -> 8 (decomp CrushUnder.cs:25).
        CardId::CrushUnder => s.base_damage = 8,
        // v0.111.0: DamageVar 30 -> 35 (decomp Devastate.cs:16).
        CardId::Devastate => s.base_damage = 35,
        // v0.111.0: DamageVar 6 -> 7 (decomp FlickFlack.cs:13).
        CardId::FlickFlack => s.base_damage = 7,
        // v0.111.0: DamageVar 16 -> 20 (decomp GiantRock.cs:14).
        CardId::GiantRock => s.base_damage = 20,
        // v0.111.0: DamageVar 16 -> 18 (decomp HowlFromBeyond.cs:15).
        CardId::HowlFromBeyond => s.base_damage = 18,
        // v0.111.0: DamageVar 15 -> 20 (decomp Mangle.cs:21).
        CardId::Mangle => s.base_damage = 20,
        // v0.111.0: DamageVar 10 -> 11 (decomp MomentumStrike.cs:16).
        CardId::MomentumStrike => s.base_damage = 11,
        // v0.111.0: DamageVar 17 -> 18 (decomp PactsEnd.cs:17).
        CardId::PactsEnd => s.base_damage = 18,
        // v0.111.0: DamageVar 9 -> 10 (decomp Rampage.cs:20).
        CardId::Rampage => s.base_damage = 10,
        // v0.111.0: DamageVar 9 -> 10 (decomp Reave.cs:17).
        CardId::Reave => s.base_damage = 10,
        // v0.111.0: DamageVar 9 -> 10 (decomp Refract.cs:25).
        CardId::Refract => s.base_damage = 10,
        // v0.111.0: DamageVar 24 -> 26 (decomp Sunder.cs:19).
        CardId::Sunder => s.base_damage = 26,
        // v0.111.0: DamageVar 28 -> 24 (decomp Hyperbeam.cs:22). The
        // HyperbeamFocusDownPower rework is S8a2's (behaviour row).
        CardId::Hyperbeam => s.base_damage = 24,

        // ── Block ───────────────────────────────────────────────────────
        // v0.111.0: BlockVar 5 -> 4 (decomp Colossus.cs:23).
        CardId::ColossusCard => s.base_block = 4,
        // v0.111.0: BlockVar 8 -> 7 (decomp MinionSacrifice.cs:19).
        CardId::MinionSacrifice => s.base_block = 7,
        // v0.111.0: BlockVar 3 -> 2 (decomp PillarOfCreation.cs:17).
        CardId::PillarOfCreation => s.base_block = 2,
        // v0.111.0: BlockVar 15 -> 16 (decomp Relax.cs:19).
        CardId::Relax => s.base_block = 16,
        // v0.111.0: BlockVar 2 -> 3 (decomp Shroud.cs:21).
        CardId::Shroud => s.base_block = 3,
        // v0.111.0: BlockVar 7 -> 6 and Uncommon -> Common (decomp
        // Taunt.cs:20,27).
        CardId::Taunt => {
            s.base_block = 6;
            s.rarity = R::Common;
        }

        // ── Keywords ────────────────────────────────────────────────────
        // v0.111.0: loses Exhaust (decomp LegionOfBone.cs:16).
        CardId::LegionOfBone => s.keywords = s.keywords.without(CardKeywords::EXHAUST),
        // v0.111.0: gains Exhaust (decomp Shatter.cs:16).
        CardId::Shatter => s.keywords = s.keywords.with(CardKeywords::EXHAUST),
        // v0.111.0: loses Exhaust (decomp Synchronize.cs:23).
        CardId::Synchronize => s.keywords = s.keywords.without(CardKeywords::EXHAUST),
        // v0.111.0: loses Exhaust (decomp TimesUp.cs:17).
        CardId::TimesUp => s.keywords = s.keywords.without(CardKeywords::EXHAUST),
        // v0.111.0: gains Exhaust (decomp Eidolon.cs:15). The auto-play
        // rework is S8a2's (behaviour row).
        CardId::Eidolon => s.keywords = s.keywords.with(CardKeywords::EXHAUST),

        _ => return None,
    }
    Some(s)
}

/// The spec-level upgrade delta of `card_id` in game build `version`
/// (absolute upgraded values, like
/// [`super::card_upgrades_generated::upgrade_delta`]).
#[must_use]
pub fn upgrade_delta_for(card_id: CardId, version: GameVersion) -> Option<CardUpgradeDelta> {
    if is_v0111(version) {
        if let Some(delta) = v0111_upgrade_delta(card_id) {
            return delta;
        }
    }
    super::card_upgrades_generated::upgrade_delta(card_id)
}

/// `Some(delta)` for a card whose spec-level v0.111.0 upgrade differs from
/// v0.107.1 (the inner `None` = no spec-level upgrade at all), `None` when
/// the v0.107.1 table entry still holds.
#[allow(clippy::unnecessary_wraps)]
fn v0111_upgrade_delta(card_id: CardId) -> Option<Option<CardUpgradeDelta>> {
    #[allow(clippy::unnecessary_wraps)]
    const fn damage(d: i32) -> Option<CardUpgradeDelta> {
        Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(d),
            base_block: None,
            keywords: None,
        })
    }
    #[allow(clippy::unnecessary_wraps)]
    const fn block(b: i32) -> Option<CardUpgradeDelta> {
        Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(b),
            keywords: None,
        })
    }
    #[allow(clippy::unnecessary_wraps)]
    const fn cost(c: i8) -> Option<CardUpgradeDelta> {
        Some(CardUpgradeDelta {
            cost: Some(c),
            base_damage: None,
            base_block: None,
            keywords: None,
        })
    }
    #[allow(clippy::unnecessary_wraps)]
    const fn keywords(k: CardKeywords) -> Option<CardUpgradeDelta> {
        Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(k),
        })
    }
    Some(match card_id {
        // Damage: new base + the (possibly new) `UpgradeValueBy` amount.
        // v0.111.0: 10 + 4 (decomp CollisionCourse.cs:36).
        CardId::CollisionCourse => damage(14),
        // v0.111.0: 8 + 1 (decomp CrushUnder.cs:51).
        CardId::CrushUnder => damage(9),
        // v0.111.0: 35 + 10 (decomp Devastate.cs:33).
        CardId::Devastate => damage(45),
        // v0.111.0: 7 + 2 (decomp FlickFlack.cs:32).
        CardId::FlickFlack => damage(9),
        // v0.111.0: 20 + 4 (decomp GiantRock.cs:31).
        CardId::GiantRock => damage(24),
        // v0.111.0: 18 + 6, was +5 (decomp HowlFromBeyond.cs:43).
        CardId::HowlFromBeyond => damage(24),
        // v0.111.0: 20 + 6, was +5 (decomp Mangle.cs:45).
        CardId::Mangle => damage(26),
        // v0.111.0: 11 + 4, was +3 (decomp MomentumStrike.cs:34).
        CardId::MomentumStrike => damage(15),
        // v0.111.0: 18 + 6 (decomp PactsEnd.cs:44).
        CardId::PactsEnd => damage(24),
        // v0.111.0: 10 + 3, was +2 (decomp Reave.cs:47).
        CardId::Reave => damage(13),
        // v0.111.0: 10 + 3 (decomp Refract.cs:49).
        CardId::Refract => damage(13),
        // v0.111.0: 26 + 8 (decomp Sunder.cs:43).
        CardId::Sunder => damage(34),
        // v0.111.0: 24 + 6, was +8 (decomp Hyperbeam.cs:61).
        CardId::Hyperbeam => damage(30),
        // Block.
        // v0.111.0: 4 + 3 (decomp Colossus.cs:41).
        CardId::ColossusCard => block(7),
        // v0.111.0: 7 + 3 (decomp MinionSacrifice.cs:33).
        CardId::MinionSacrifice => block(10),
        // v0.111.0: 2 + 1 (decomp PillarOfCreation.cs:31).
        CardId::PillarOfCreation => block(3),
        // v0.111.0: 16 + 2 (decomp Relax.cs:43).
        CardId::Relax => block(18),
        // v0.111.0: 3 + 1 (decomp Shroud.cs:36).
        CardId::Shroud => block(4),
        // v0.111.0: 6 + 1 (decomp Taunt.cs:41).
        CardId::Taunt => block(7),
        // Cost / keywords.
        // v0.111.0: upgrade removes Exhaust; v0.107.1 cost 1 less (decomp
        // Mirage.cs:42).
        CardId::Mirage => keywords(CardKeywords::NONE),
        // v0.111.0: upgrade costs 1 less; v0.107.1 gained Innate (decomp
        // TrashToTreasure.cs:28).
        CardId::TrashToTreasure => cost(0),
        // v0.111.0: upgrade costs 1 less (2 -> 1); v0.107.1 raised the
        // RetainAmount var (decomp WellLaidPlans.cs:18).
        CardId::WellLaidPlans => cost(1),
        // v0.111.0: the upgrade no longer lowers the cost — it raises the
        // Block vars, which S8a2's rework reads (decomp ExpectAFight.cs:39).
        CardId::ExpectAFight => None,
        // v0.111.0: Exhaust is no longer a base keyword, so the upgrade's
        // Retain leaves Retain alone (decomp TimesUp.cs:41).
        CardId::TimesUp => keywords(CardKeywords::RETAIN),
        // v0.111.0: no longer Exhaust, and the upgrade raises the
        // per-orb extra instead of removing Exhaust (decomp Synchronize.cs:41).
        CardId::Synchronize => None,
        // v0.111.0: the base gained Exhaust; the upgrade still costs 1 less
        // (decomp Eidolon.cs:36), which the v0.107.1 table already says.
        _ => return None,
    })
}

/// The card that holds `card_id`'s pool slot in game build `version`, or
/// `None` when the slot is unchanged. v0.111.0 deleted Scare
/// (`Models.Cards/Scare.cs` is gone) and `SilentCardPool.cs:68` lists
/// Sidestep exactly where v0.107.1 listed Scare (no card sorts between them
/// in either build), so a v0.111.0 pool walk swaps it in place and every
/// `NextItem` index over the pool keeps C#'s order.
#[must_use]
pub fn pool_substitute(card_id: CardId, version: GameVersion) -> Option<fn() -> CardSpec> {
    match card_id {
        CardId::Scare if is_v0111(version) => Some(super::silent::sidestep::spec),
        _ => None,
    }
}

/// The spec in `factory`'s pool slot as game build `version` defines it:
/// [`pool_substitute`] then [`spec_for`]. Exactly `factory()` for v0.107.1.
#[must_use]
pub fn pool_spec(factory: fn() -> CardSpec, version: GameVersion) -> CardSpec {
    pool_spec_of(factory(), version)
}

/// [`pool_spec`] over an already-built spec (the combat-generation pools
/// hold `CardSpec`s, not factories).
#[must_use]
pub fn pool_spec_of(spec: CardSpec, version: GameVersion) -> CardSpec {
    let spec = pool_substitute(spec.id, version).map_or(spec, |f| f());
    spec_for(spec, version)
}

/// v0.111.0 `CanBeGeneratedInCombat => false` additions. Composes with the
/// per-pool v0.107.1 filters (which already drop Alchemize, Feed, …):
/// returns `false` only for the cards this build newly excludes.
#[must_use]
pub fn can_be_generated_in_combat(card_id: CardId, version: GameVersion) -> bool {
    !(is_v0111(version)
        && matches!(
            card_id,
            // v0.111.0: decomp AscendersBane.cs:10, Eidolon.cs:13,
            // Nightmare.cs:21, Transfigure.cs:14, Abundance.cs:13 (Abundance
            // is v0.111.0-only and sits in the EventCardPool).
            CardId::AscendersBane
                | CardId::Eidolon
                | CardId::Nightmare
                | CardId::Transfigure
                | CardId::Abundance
        ))
}

/// Whether `card_id`'s POD or upgrade differs in game build `version` from
/// v0.107.1. Rarity-only changes do not count: rarity is not on the POD.
#[must_use]
pub fn pod_changes_in(card_id: CardId, version: GameVersion) -> bool {
    is_v0111(version) && v0111_pods().contains_key(&(card_id, false))
}

/// Whether `card_id` changed at the POD level in v0.111.0 — uncached; see
/// [`v0111_pods`].
fn pod_changes_v0111(card_id: CardId) -> bool {
    if v0111_upgrade_delta(card_id).is_some() {
        return true;
    }
    let Some(old) = super::spec_index::spec_for_card_id(card_id) else {
        return false;
    };
    let new = spec_for(old, GameVersion::V0_111_0);
    (
        old.cost,
        old.card_type,
        old.target_type,
        old.base_damage,
        old.base_block,
        old.keywords,
    ) != (
        new.cost,
        new.card_type,
        new.target_type,
        new.base_damage,
        new.base_block,
        new.keywords,
    )
}

/// The v0.111.0 PODs of every card whose POD changed, keyed by
/// `(card_id, upgraded)`. A pure function of the static spec tables, built
/// once: [`card_pod_for`] sits on the combat spawn path through
/// `Card::normalize_for_version`.
fn v0111_pods() -> &'static std::collections::BTreeMap<(CardId, bool), sts2_core::cards::Card> {
    use std::collections::BTreeMap;
    use std::sync::OnceLock;
    static PODS: OnceLock<BTreeMap<(CardId, bool), sts2_core::cards::Card>> = OnceLock::new();
    PODS.get_or_init(|| {
        use super::spec_index as si;
        let slices: [&[fn() -> CardSpec]; 8] = [
            si::IRONCLAD_SPECS,
            si::SILENT_SPECS,
            si::DEFECT_SPECS,
            si::REGENT_SPECS,
            si::NECROBINDER_SPECS,
            si::COLORLESS_SPECS,
            si::STATUS_SPECS,
            si::V0111_ONLY_SPECS,
        ];
        let mut map = BTreeMap::new();
        for slice in slices {
            for factory in slice {
                let id = factory().id;
                if map.contains_key(&(id, false)) || !pod_changes_v0111(id) {
                    continue;
                }
                for upgraded in [false, true] {
                    if let Some(card) = build_v0111_pod(id, upgraded) {
                        map.insert((id, upgraded), card);
                    }
                }
            }
        }
        map
    })
}

/// Build the v0.111.0 POD of a changed card. The keyword override is set
/// only when the base keywords differ from v0.107.1's (the engine's
/// `CardId`-keyed keyword registries hold the v0.107.1 set), so it does not
/// depend on whether those registries are installed yet.
fn build_v0111_pod(card_id: CardId, upgraded: bool) -> Option<sts2_core::cards::Card> {
    let old = super::spec_index::spec_for_card_id(card_id)?;
    let spec = spec_for(old, GameVersion::V0_111_0);
    let mut card = super::status::card_from_spec(&spec);
    if spec.keywords != old.keywords {
        card.keyword_override = Some(super::spec_index::keyword_override_from(spec.keywords));
    }
    card.upgraded = upgraded;
    if upgraded {
        if let Some(delta) = upgrade_delta_for(card_id, GameVersion::V0_111_0) {
            super::spec_index::apply_upgrade_delta(&mut card, delta);
        }
    }
    Some(card)
}

/// The card POD for `(card_id, upgraded)` in game build `version`.
///
/// v0.107.1, and any card whose POD did not change, is exactly
/// [`super::spec_index::card_pod_from_id`]. A changed card is built from
/// [`spec_for`] + [`upgrade_delta_for`]; a keyword change on the un-upgraded
/// card is carried on `keyword_override`. Registered with core as the
/// versioned card factory.
#[must_use]
pub fn card_pod_for(
    card_id: CardId,
    upgraded: bool,
    version: GameVersion,
) -> Option<sts2_core::cards::Card> {
    if is_v0111(version) {
        if let Some(card) = v0111_pods().get(&(card_id, upgraded)) {
            return Some(*card);
        }
    }
    super::spec_index::card_pod_from_id(card_id, upgraded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pod(id: CardId, upgraded: bool, v: GameVersion) -> sts2_core::cards::Card {
        crate::cards::dispatch::ensure_installed();
        card_pod_for(id, upgraded, v).expect("spec exists")
    }

    #[test]
    fn v0107_pod_is_card_pod_from_id() {
        crate::cards::dispatch::ensure_installed();
        for id in [
            CardId::Reave,
            CardId::Mangle,
            CardId::Haze,
            CardId::Shatter,
            CardId::Bash,
        ] {
            for up in [false, true] {
                assert_eq!(
                    card_pod_for(id, up, GameVersion::V0_107_1),
                    crate::cards::spec_index::card_pod_from_id(id, up),
                    "{id:?} upgraded={up}"
                );
            }
        }
    }

    #[test]
    fn reave_damage_v0111() {
        assert_eq!(
            pod(CardId::Reave, false, GameVersion::V0_107_1).base_damage,
            9
        );
        assert_eq!(
            pod(CardId::Reave, true, GameVersion::V0_107_1).base_damage,
            11
        );
        assert_eq!(
            pod(CardId::Reave, false, GameVersion::V0_111_0).base_damage,
            10
        );
        assert_eq!(
            pod(CardId::Reave, true, GameVersion::V0_111_0).base_damage,
            13
        );
    }

    #[test]
    fn upgrade_for_uses_the_build_delta_v0111() {
        crate::cards::dispatch::ensure_installed();
        let mut c = pod(CardId::Mangle, false, GameVersion::V0_111_0);
        assert_eq!(c.base_damage, 20);
        assert!(c.upgrade_for(GameVersion::V0_111_0));
        assert_eq!(c.base_damage, 26);
        assert!(c.downgrade_for(GameVersion::V0_111_0));
        assert_eq!(c.base_damage, 20);
    }

    #[test]
    fn keyword_changes_ride_keyword_override_v0111() {
        use sts2_core::card_effects::{card_exhausts_inst, card_innate_inst, card_retain_inst};
        let v = GameVersion::V0_111_0;
        assert!(card_exhausts_inst(&pod(CardId::Shatter, false, v)));
        assert!(!card_exhausts_inst(&pod(
            CardId::Shatter,
            false,
            GameVersion::V0_107_1
        )));
        assert!(!card_exhausts_inst(&pod(CardId::LegionOfBone, false, v)));
        assert!(!card_exhausts_inst(&pod(CardId::Synchronize, false, v)));
        assert!(!card_exhausts_inst(&pod(CardId::Synchronize, true, v)));
        let times_up = pod(CardId::TimesUp, true, v);
        assert!(!card_exhausts_inst(&times_up) && card_retain_inst(&times_up));
        // Mirage keeps Exhaust and its cost; the upgrade drops Exhaust.
        let mirage = pod(CardId::Mirage, true, v);
        assert_eq!(mirage.local_cost(), 1);
        assert!(!card_exhausts_inst(&mirage));
        assert!(card_exhausts_inst(&pod(CardId::Mirage, false, v)));
        // TrashToTreasure upgrades to cost 0, not Innate.
        let ttt = pod(CardId::TrashToTreasure, true, v);
        assert_eq!(ttt.local_cost(), 0);
        assert!(!card_innate_inst(&ttt));
    }

    #[test]
    fn outbreak_is_a_rare_skill_v0111() {
        let s = spec_for(
            crate::cards::silent::outbreak::spec(),
            GameVersion::V0_111_0,
        );
        assert_eq!(s.card_type, CardType::Skill);
        assert_eq!(s.rarity, CardRarity::Rare);
        assert_eq!(s.cost, 3);
        assert_eq!(s.target_type, TargetType::AllEnemies);
    }

    #[test]
    fn rarity_is_not_a_pod_change() {
        assert!(!pod_changes_in(CardId::Bloodletting, GameVersion::V0_111_0));
        assert!(pod_changes_in(CardId::Taunt, GameVersion::V0_111_0));
        assert!(!pod_changes_in(CardId::Taunt, GameVersion::V0_107_1));
    }

    #[test]
    fn sidestep_takes_scares_slot_only_v0111() {
        assert!(pool_substitute(CardId::Scare, GameVersion::V0_111_0).is_some());
        assert!(pool_substitute(CardId::Scare, GameVersion::V0_107_1).is_none());
        assert_eq!(
            pool_spec(crate::cards::silent::scare::spec, GameVersion::V0_111_0).id,
            CardId::Sidestep
        );
        assert_eq!(
            pool_spec(crate::cards::silent::scare::spec, GameVersion::V0_107_1).id,
            CardId::Scare
        );
    }

    /// A one-enemy combat with `card_id` (built in `version`) on the Play pile.
    fn combat_with(
        card_id: CardId,
        upgraded: bool,
        version: GameVersion,
    ) -> (sts2_core::CombatState, sts2_core::CreatureUid) {
        crate::cards::dispatch::ensure_installed();
        let mut state = sts2_core::CombatState {
            game_version: version,
            ..Default::default()
        };
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let enemy = state.spawn_enemy(sts2_core::MonsterKind::Nibbit);
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
            c.current_hp = 99;
            c.max_hp = 99;
        }
        let mut card = card_pod_for(card_id, upgraded, version).expect("spec exists");
        let uid = sts2_core::CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits"))
            .expect("nz");
        card.uid = uid;
        state.cards.push(card);
        state.player.play.push(uid);
        (state, enemy)
    }

    fn play(state: &mut sts2_core::CombatState, id: CardId, target: sts2_core::CreatureUid) {
        let f = sts2_core::card_effects::lookup(id).expect("registered");
        let _ = f(state, Some(target));
    }

    #[test]
    fn haze_also_applies_weak_v0111() {
        for (version, weak) in [(GameVersion::V0_107_1, 0), (GameVersion::V0_111_0, 1)] {
            let (mut state, enemy) = combat_with(CardId::Haze, false, version);
            play(&mut state, CardId::Haze, enemy);
            assert_eq!(
                crate::cards::colorless::helpers::power_amount_on_creature(
                    &state,
                    enemy,
                    sts2_core::PowerKind::Poison
                ),
                4
            );
            assert_eq!(
                crate::cards::colorless::helpers::power_amount_on_creature(
                    &state,
                    enemy,
                    sts2_core::PowerKind::Weak
                ),
                weak,
                "{version}"
            );
        }
        assert!(sts2_core::card_effects::card_sly_in(
            CardId::Haze,
            GameVersion::V0_107_1
        ));
        assert!(!sts2_core::card_effects::card_sly_in(
            CardId::Haze,
            GameVersion::V0_111_0
        ));
    }

    #[test]
    fn thunder_and_demon_form_amounts_v0111() {
        for (version, upgraded, want) in [
            (GameVersion::V0_107_1, false, 6),
            (GameVersion::V0_107_1, true, 8),
            (GameVersion::V0_111_0, false, 8),
            (GameVersion::V0_111_0, true, 11),
        ] {
            let (mut state, enemy) = combat_with(CardId::ThunderCard, upgraded, version);
            play(&mut state, CardId::ThunderCard, enemy);
            assert_eq!(
                state.player_power_amount(sts2_core::PowerKind::Thunder),
                want
            );
        }
        for (version, want) in [(GameVersion::V0_107_1, 2), (GameVersion::V0_111_0, 3)] {
            let (mut state, enemy) = combat_with(CardId::DemonFormCard, false, version);
            play(&mut state, CardId::DemonFormCard, enemy);
            assert_eq!(
                state.player_power_amount(sts2_core::PowerKind::DemonForm),
                want
            );
        }
    }

    #[test]
    fn fuel_no_longer_draws_v0111() {
        for (version, upgraded, energy, hand) in [
            (GameVersion::V0_107_1, false, 1, 1),
            (GameVersion::V0_107_1, true, 1, 2),
            (GameVersion::V0_111_0, false, 1, 0),
            (GameVersion::V0_111_0, true, 2, 0),
        ] {
            let (mut state, enemy) = combat_with(CardId::Fuel, upgraded, version);
            for _ in 0..3 {
                let mut c = card_pod_for(CardId::StrikeSilent, false, version).expect("spec");
                let uid =
                    sts2_core::CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits"))
                        .expect("nz");
                c.uid = uid;
                state.cards.push(c);
                state.player.draw.push(uid);
            }
            state.player.energy = 0;
            play(&mut state, CardId::Fuel, enemy);
            assert_eq!(state.player.energy, energy, "{version} upgraded={upgraded}");
            assert_eq!(
                state.player.hand.len(),
                hand,
                "{version} upgraded={upgraded}"
            );
        }
    }

    #[test]
    fn star_costs_v0111() {
        let (state, _) = combat_with(CardId::Alignment, false, GameVersion::V0_111_0);
        let card = state.cards[0];
        assert_eq!(sts2_core::effective_star_cost_in_combat(&state, &card), 2);
        let (state, _) = combat_with(CardId::Alignment, false, GameVersion::V0_107_1);
        let card = state.cards[0];
        assert_eq!(sts2_core::effective_star_cost_in_combat(&state, &card), 3);
        let (state, _) = combat_with(CardId::GuidingStar, false, GameVersion::V0_111_0);
        let card = state.cards[0];
        assert_eq!(sts2_core::effective_star_cost_in_combat(&state, &card), 1);
    }

    #[test]
    fn pools_read_the_build_rarity_v0111() {
        use crate::cards::spec_index::{rarity_for_card_id_in, specs_in, SILENT_SPECS};
        assert_eq!(
            rarity_for_card_id_in(CardId::Bloodletting, GameVersion::V0_107_1),
            Some(CardRarity::Common)
        );
        assert_eq!(
            rarity_for_card_id_in(CardId::Bloodletting, GameVersion::V0_111_0),
            Some(CardRarity::Uncommon)
        );
        let ids = |v| specs_in(SILENT_SPECS, v).map(|s| s.id).collect::<Vec<_>>();
        assert!(ids(GameVersion::V0_107_1).contains(&CardId::Scare));
        assert!(!ids(GameVersion::V0_111_0).contains(&CardId::Scare));
        // SilentCardPool.cs:68 (v0.111.0): Sidestep in Scare's slot.
        let (old, new) = (ids(GameVersion::V0_107_1), ids(GameVersion::V0_111_0));
        assert_eq!(old.len(), new.len());
        let i = old.iter().position(|&c| c == CardId::Scare).unwrap();
        assert_eq!(new[i], CardId::Sidestep);
    }

    #[test]
    fn combat_pool_drops_scare_and_non_generatable_v0111() {
        use crate::cards::colorless::helpers::combat_generatable_pool_in;
        use sts2_core::potions::CharacterId;
        let ids = |v| {
            combat_generatable_pool_in(CharacterId::Silent, v)
                .iter()
                .map(|s| s.id)
                .collect::<Vec<_>>()
        };
        let old = ids(GameVersion::V0_107_1);
        let new = ids(GameVersion::V0_111_0);
        assert!(old.contains(&CardId::Scare) && old.contains(&CardId::Nightmare));
        assert!(!new.contains(&CardId::Scare) && !new.contains(&CardId::Nightmare));
        // Sidestep takes Scare's slot: same neighbours on both sides.
        let i = old.iter().position(|&c| c == CardId::Scare).unwrap();
        let j = new.iter().position(|&c| c == CardId::Sidestep).unwrap();
        assert_eq!(old[i - 1], new[j - 1]);
        assert_eq!(old[i + 1], new[j + 1]);
        assert!(
            !new.contains(&CardId::OutbreakCard) || {
                let ob = combat_generatable_pool_in(CharacterId::Silent, GameVersion::V0_111_0)
                    .iter()
                    .find(|s| s.id == CardId::OutbreakCard)
                    .unwrap();
                ob.card_type == CardType::Skill
            }
        );
        let necro = combat_generatable_pool_in(CharacterId::Necrobinder, GameVersion::V0_111_0);
        assert!(!necro
            .iter()
            .any(|s| matches!(s.id, CardId::Eidolon | CardId::Transfigure)));
    }

    #[test]
    fn normalize_moves_a_v0107_pod_once_v0111() {
        crate::cards::dispatch::ensure_installed();
        let v = GameVersion::V0_111_0;
        let mut c = crate::cards::spec_index::card_pod_from_id(CardId::Reave, true).expect("spec");
        c.normalize_for_version(v);
        assert_eq!(c.base_damage, 13);
        c.normalize_for_version(v);
        assert_eq!(c.base_damage, 13, "idempotent");
        // A v0.107.1 base card upgraded through the build's path lands on
        // the build's upgraded value.
        let mut c = crate::cards::spec_index::card_pod_from_id(CardId::Reave, false).expect("spec");
        assert!(c.upgrade_for(v));
        assert_eq!(c.base_damage, 13);
        // v0.107.1 is untouched.
        let mut c = crate::cards::spec_index::card_pod_from_id(CardId::Reave, false).expect("spec");
        c.normalize_for_version(GameVersion::V0_107_1);
        assert_eq!(c.base_damage, 9);
    }
}
