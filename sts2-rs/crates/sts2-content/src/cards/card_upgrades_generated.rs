//! Spec-level upgrade deltas (cost / base_damage / base_block /
//! keywords) per card. `effect_vars`-only upgrades are handled inline
//! in the effect handlers and intentionally absent here.
//!
//! HAND-MAINTAINED against the C# decompilation (v0.103.2), the source
//! of truth for upgrade values (base + `UpgradeValueBy`). The former
//! scrape_card_upgrades.py generator was removed. Bead sts2-rs-vd6pf.

use sts2_core::cards::CardId;

use super::types::{CardKeywords, CardUpgradeDelta};

/// Spec-level upgraded stats for `card_id`, or `None` if upgrading the
/// card leaves cost/damage/block/keywords unchanged (effect-only).
#[must_use]
pub fn upgrade_delta(card_id: CardId) -> Option<CardUpgradeDelta> {
    match card_id {
        // Bead `sts2-rs-u5o3o`: C# `Anointed.cs` OnUpgrade adds the Retain
        // keyword (`AddKeyword(Retain)`); spec is EXHAUST so the absolute
        // upgraded set is EXHAUST + RETAIN.
        CardId::Apotheosis => Some(CardUpgradeDelta {
            cost: Some(1i8),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        CardId::Apparition => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(CardKeywords::EXHAUST),
        }),
        // Bead `sts2-rs-giozp`: C# `Arsenal.cs` OnUpgrade
        // `AddKeyword(Innate)`; spec NONE → INNATE.
        // Bead `sts2-rs-af3h4`: C# `Automation.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)` 1 → 0.
        // Bead `sts2-rs-eiznx`: C# `BansheesCry.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-2)`; base cost 9 → 7.
        CardId::BansheesCry => Some(CardUpgradeDelta {
            cost: Some(7i8),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-giozp`: C# `BeaconOfHope.cs` OnUpgrade
        // `AddKeyword(Innate)`; spec NONE → INNATE.
        // Bead `sts2-rs-tk230`: C# `BlightStrike.cs` damage 8 → 10.
        CardId::BlightStrike => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(10),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-z13x4`: C# `Bombardment.cs` damage 18 → 24.
        // Bead `sts2-rs-h4ect`: C# `Bury.cs` `DamageVar(52m)` with
        // `OnUpgrade UpgradeValueBy(11m)` → 52 base / 63 upgraded.
        CardId::Bury => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(63),
            base_block: None,
            keywords: None,
        }),
        CardId::ByrdSwoop => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(18),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-uevkk`: C# `CalculatedGamble.cs` OnUpgrade adds the
        // Retain keyword (`AddKeyword(CardKeyword.Retain)`); spec is EXHAUST
        // so the absolute upgraded set is EXHAUST + RETAIN.
        // Bead `sts2-rs-nidl4`: C# `CallOfTheVoid.cs` OnUpgrade
        // `AddKeyword(CardKeyword.Innate)`; spec NONE → INNATE.
        CardId::CallOfTheVoid => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(CardKeywords::INNATE),
        }),
        // Bead `sts2-rs-tk230`: C# `CaptureSpirit.cs` damage 3 → 4.
        CardId::CaptureSpirit => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(4),
            base_block: None,
            keywords: None,
        }),
        CardId::Clash => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(18),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-z13x4`: C# `Comet.cs` damage 33 → 44.
        // Bead `sts2-rs-lftlf`: C# `Compact.cs` block 6 → 7.
        // Bead `sts2-rs-x1xxa`: C# `Conflagration.cs` OnUpgrade bumps
        // CalculationBase 8→9 (the spec-side base_damage). ExtraDamage
        // per-Attack scaling 2→3 is applied in the handler via
        // `current_upgraded(state)`.
        // Bead `sts2-rs-ki1vz`: C# `CrushUnder.cs` DamageVar 7 → 8.
        // C# `DeathMarch.cs` OnUpgrade `CalculationBase.UpgradeValueBy(1)`
        // bumps the spec-side CalculationBase 8 → 9 (the `base_damage` here).
        // The `ExtraDamage` per-non-hand-draw scaling 3 → 4 is effect-vars-only
        // and applied in `play_death_march` via `current_card_upgraded(state)`,
        // not a `base_damage` stat.
        CardId::DeathMarch => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(9),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-t8b3k`: C# `DeathsDoor.cs` block 6 → 7.
        CardId::DeathsDoor => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(7),
            keywords: None,
        }),
        // C# `Debilitate.cs` OnUpgrade `Damage.UpgradeValueBy(2)` bumps the spec
        // base damage 10 → 12. The `DebilitatePower` amount 3 → 4 is an
        // effect-var bump applied in `play_debilitate` via
        // `current_card_upgraded(state)`, not a `base_damage` stat.
        CardId::DebilitateCard => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(12),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-hg7w1`: C# `DefendNecrobinder.cs` OnUpgrade
        // `Block.UpgradeValueBy(3m)` ⇒ base 5 / upgrade 8. Systemic
        // upgrade-gap vein `sts2-rs-vd6pf`.
        CardId::DefendNecrobinder => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(8),
            keywords: None,
        }),
        // Bead `sts2-rs-tk230`: C# `Defile.cs` damage 13 → 17.
        CardId::Defile => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(17),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-t8b3k`: C# `Defy.cs` block 6 → 9.
        CardId::Defy => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(9),
            keywords: None,
        }),
        // Bead `sts2-rs-t8b3k`: C# `Delay.cs` block 11 → 13.
        CardId::Delay => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(13),
            keywords: None,
        }),
        // Bead `sts2-rs-eiznx`: C# `Demesne.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)`; base cost 3 → 2.
        CardId::Demesne => Some(CardUpgradeDelta {
            cost: Some(2i8),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-z13x4`: C# `Devastate.cs` damage 30 → 40.
        CardId::Distraction => Some(CardUpgradeDelta {
            cost: Some(0i8),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-tk230`: C# `DrainPower.cs` damage 10 → 12.
        CardId::DrainPower => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(12),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-nidl4`: C# `Dredge.cs` OnUpgrade
        // `AddKeyword(CardKeyword.Retain)`; spec EXHAUST → EXHAUST + RETAIN.
        CardId::Dredge => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(CardKeywords::EXHAUST.with(CardKeywords::RETAIN)),
        }),
        // C# `DyingStar.cs` OnUpgrade: Damage 9→11 (+2); StrengthLoss
        // is `effect_vars`-only and gated inline in `play_dying_star`.
        // Bead `sts2-rs-upqp8`.
        // Bead `sts2-rs-eiznx`: C# `Eidolon.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)`; base cost 2 → 1.
        CardId::Eidolon => Some(CardUpgradeDelta {
            cost: Some(1i8),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        CardId::Entrench => Some(CardUpgradeDelta {
            cost: Some(1i8),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-tk230`: C# `Eradicate.cs` damage 11 → 14.
        CardId::Eradicate => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(14),
            base_block: None,
            keywords: None,
        }),
        CardId::Exterminate => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(4),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-tk230`: C# `Fear.cs` damage 7 → 8.
        CardId::Fear => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(8),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-z13x4`: C# `FlakCannon.cs` damage 8 → 11.
        // Bead `sts2-rs-eiznx`: C# `ForbiddenGrimoire.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)`; base cost 2 → 1.
        CardId::ForbiddenGrimoire => Some(CardUpgradeDelta {
            cost: Some(1i8),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-z13x4`: C# `GammaBlast.cs` damage 13 → 18.
        CardId::GiantRock => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(20),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-ki1vz`: C# `Glitterstream.cs` BlockVar 11 → 13.
        // Bead `sts2-rs-k3edn`: C# `GoldAxe.cs` OnUpgrade
        // `AddKeyword(Retain)`; spec NONE → RETAIN.
        // Bead `sts2-rs-bq80l`: C# `Graveblast.cs` OnUpgrade bumps damage
        // 4→6 AND `RemoveKeyword(Exhaust)`. Spec keywords are EXHAUST so
        // the upgraded absolute set is NONE.
        CardId::Graveblast => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(6),
            base_block: None,
            keywords: Some(CardKeywords::NONE),
        }),
        // Bead `sts2-rs-t8b3k`: C# `GraveWarden.cs` block 8 → 11.
        CardId::GraveWarden => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(11),
            keywords: None,
        }),
        // Bead `sts2-rs-k6rp7`: C# `HammerTime.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)` 2 → 1.
        // Bead `sts2-rs-tk230`: C# `Hang.cs` damage 10 → 13.
        CardId::Hang => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(13),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-z13x4`: C# `HeavenlyDrill.cs` damage 8 → 10.
        // Bead `sts2-rs-9hp50`: C# `Hegemony.cs` DamageVar 15 → 18
        // (`OnUpgrade: Damage.UpgradeValueBy(3m)`).
        CardId::HelloWorldCard => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(CardKeywords::INNATE),
        }),
        // sts2-rs-oq6nf: C# `Hologram.cs` OnUpgrade bumps Block 3 -> 5 and
        // `RemoveKeyword(Exhaust)` -> upgraded Hologram is no longer Exhaust.
        // Bead `sts2-rs-76kjk`: C# `Hotfix.cs` OnUpgrade `RemoveKeyword(Exhaust)`
        // ⇒ upgraded Hotfix has no Exhaust keyword. No damage/block/cost delta.
        // Bead `sts2-rs-lftlf`: C# `IAmInvincible.cs` block 10 → 13.
        // Bead `sts2-rs-z13x4`: C# `KinglyKick.cs` damage 27 → 35.
        // Bead `sts2-rs-ki1vz`: C# `KinglyPunch.cs` DamageVar 8 → 10.
        // Bead `sts2-rs-1yi26`: C# `Knockdown.cs OnUpgrade`
        // `DynamicVars.Damage.UpgradeValueBy(4m)` → 10 → 14. The
        // sibling `KnockdownPower` 2 → 3 bump is gated inline in
        // `play_knockdown` via `current_upgraded`.
        // Bead `sts2-rs-z13x4`: C# `KnockoutBlow.cs` damage 30 → 38.
        // Bead `sts2-rs-z13x4`: C# `LunarBlast.cs` damage 4 → 5.
        CardId::MadScience => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(CardKeywords::INNATE),
        }),
        CardId::Maul => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(6),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-t8b3k`: C# `Melancholy.cs` block 13 → 17.
        CardId::Melancholy => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(17),
            keywords: None,
        }),
        // Bead `sts2-rs-z13x4`: C# `MeteorShower.cs` damage 14 → 21.
        // Bead `sts2-rs-giozp`: C# `Mimic.cs` OnUpgrade
        // `RemoveKeyword(Exhaust)`; spec EXHAUST → NONE.
        // Bead `sts2-rs-pc0h1`: C# `MindBlast.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)` 1 → 0. Damage is a CalculatedVar
        // (draw-pile count) and does not upgrade.
        CardId::MinionDiveBomb => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(16),
            base_block: None,
            keywords: None,
        }),
        CardId::MinionSacrifice => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(11),
            keywords: None,
        }),
        CardId::MinionStrike => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(9),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-3ebox`: C# `Misery.cs` OnUpgrade Damage 7→9 +
        // AddKeyword(Retain); no base keywords → upgraded kw = RETAIN.
        CardId::Misery => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(9),
            base_block: None,
            keywords: Some(CardKeywords::RETAIN),
        }),
        // Bead `sts2-rs-k6rp7`: C# `MonarchsGaze.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)` 3 → 2.
        // Bead `sts2-rs-giozp`: C# `Monologue.cs` OnUpgrade
        // `AddKeyword(Retain)`; spec NONE → RETAIN.
        CardId::NeowsFury => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(14),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-faaku`: C# `NegativePulse.cs` OnUpgrade bumps
        // BlockVar +1 (5→6). The Doom power amount 7→11 is handled in
        // the handler via `current_upgraded(state)`.
        CardId::NegativePulse => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(6),
            keywords: None,
        }),
        // Bead `sts2-rs-af3h4`: C# `Omnislice.cs` DamageVar 8 → 11
        // (`OnUpgrade Damage.UpgradeValueBy(3)`). play_omnislice resolves
        // through current_damage(state) with a max-floor of 8.
        // Bead `sts2-rs-k6rp7`: C# `Orbit.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)` 2 → 1.
        // Bead `sts2-rs-eiznx`: C# `Pagestorm.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)`; base cost 1 → 0.
        CardId::Pagestorm => Some(CardUpgradeDelta {
            cost: Some(0i8),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-lftlf`: C# `ParticleWall.cs` block 9 → 12.
        // Bead `sts2-rs-t8b3k`: C# `PillarOfCreation.cs` block 3 → 4.
        // Bead `sts2-rs-eiznx`: C# `Protector.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)`; base cost 1 → 0. (CalcBase 10→15
        // is effect-vars-only and gated by the handler, not base_damage.)
        CardId::Protector => Some(CardUpgradeDelta {
            cost: Some(0i8),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-t8b3k`: C# `PullAggro.cs` block 7 → 9.
        CardId::PullAggro => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(9),
            keywords: None,
        }),
        // Bead `sts2-rs-tk230`: C# `PullFromBelow.cs` damage 5 → 7.
        CardId::PullFromBelow => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(7),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-z13x4`: C# `Radiate.cs` damage 3 → 4.
        // Bead `sts2-rs-lftlf`: C# `Rally.cs` block 12 → 17.
        // Bead `sts2-rs-tk230`: C# `Reap.cs` damage 27 → 33.
        CardId::Reap => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(33),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-3ebox`: C# `ReaperForm.cs` OnUpgrade
        // `AddKeyword(Retain)`; no base keywords → upgraded kw = RETAIN.
        CardId::ReaperForm => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(CardKeywords::RETAIN),
        }),
        // Bead `sts2-rs-tk230`: C# `Reave.cs` damage 9 → 11.
        CardId::Reave => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(11),
            base_block: None,
            keywords: None,
        }),
        CardId::Rebound => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(12),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-448w0`: C# `Reflect.cs` Block 17 → 21
        // (`OnUpgrade: Block.UpgradeValueBy(4m)`). play_reflect_card
        // routes through current_block(), so the gap was solely the
        // missing upgrade-table entry.
        CardId::Relax => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(17),
            keywords: None,
        }),
        CardId::RipAndTear => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(9),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-giozp`: C# `RoyalGamble.cs` OnUpgrade
        // `AddKeyword(Retain)`; spec EXHAUST → EXHAUST + RETAIN.
        // Bead `sts2-rs-eiznx`: C# `Sacrifice.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)`; base cost 1 → 0.
        CardId::Sacrifice => Some(CardUpgradeDelta {
            cost: Some(0i8),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-z13x4`: C# `Salvo.cs` damage 12 → 16.
        // Bead `sts2-rs-giozp`: C# `Scrawl.cs` OnUpgrade
        // `AddKeyword(Retain)`; spec EXHAUST → EXHAUST + RETAIN.
        // Bead `sts2-rs-tk230`: C# `SculptingStrike.cs` damage 9 → 12.
        CardId::SculptingStrike => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(12),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-eiznx`: C# `Seance.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)`; base cost 1 → 0.
        CardId::Seance => Some(CardUpgradeDelta {
            cost: Some(0i8),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-t8b3k`: C# `SecondWind.cs` block 5 → 7.
        // Bead `sts2-rs-eiznx`: C# `SentryMode.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)`; base cost 2 → 1.
        CardId::SentryMode => Some(CardUpgradeDelta {
            cost: Some(1i8),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-ki1vz`: C# `SevenStars.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)` 2 → 1. Damage stays 7 (no UpgradeValueBy
        // on DamageVar).
        // Bead `sts2-rs-qnikp`: C# `Severance.cs` DamageVar 13 → 18
        // (`OnUpgrade Damage.UpgradeValueBy(5)`).
        CardId::Severance => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(18),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-z13x4`: C# `ShiningStrike.cs` damage 8 → 11.
        CardId::Shiv => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(6),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-t8b3k`: C# `Shroud.cs` block 2 → 3.
        CardId::Shroud => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(3),
            keywords: None,
        }),
        CardId::SovereignBlade => Some(CardUpgradeDelta {
            cost: Some(1i8),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-tk230`: C# `Sow.cs` damage 8 → 11.
        CardId::Sow => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(11),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-giozp`: C# `Speedster.cs` OnUpgrade
        // `AddKeyword(Innate)`; spec NONE → INNATE.
        // Bead `sts2-rs-k6rp7`: C# `SpectrumShift.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)` 2 → 1.
        CardId::Squash => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(12),
            base_block: None,
            keywords: None,
        }),
        CardId::Stack => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(3),
            keywords: None,
        }),
        // Bead `sts2-rs-z13x4`: C# `Stardust.cs` damage 5 → 7.
        // Stoke: C# `Stoke.cs` has NO `OnUpgrade` override — upgrade
        // does not change cost/damage/block/keywords. The upgrade
        // surface lives in OnPlay (each generated card gets
        // `CardCmd.Upgrade` if `IsUpgraded`). Prior entry incorrectly
        // dropped Stoke's cost 1→0. Bead `sts2-rs-tfzwk`.
        // Bead `sts2-rs-hg7w1`: C# `StrikeNecrobinder.cs` OnUpgrade
        // `Damage.UpgradeValueBy(3m)` ⇒ base 6 / upgrade 9. Systemic
        // upgrade-gap vein `sts2-rs-vd6pf`.
        CardId::StrikeNecrobinder => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(9),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-4iaxt`: C# `Sunder.cs` OnUpgrade bumps DamageVar
        // 24→32. The on-kill EnergyVar(3) rider does not upgrade and is
        // applied in the handler.
        CardId::SweepingGaze => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(15),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-k6rp7`: C# `SwordSage.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)` 2 → 1.
        // Bead `sts2-rs-lftlf`: C# `TheGambit.cs` block 50 → 75.
        // v0.107.1: C# `TheSealedThrone.cs` OnUpgrade is now
        // `EnergyCost.UpgradeBy(-1)` (cost 1 → 0); the v0.103.2 `AddKeyword(Innate)`
        // is gone. Delta delegates to `the_sealed_throne::upgrade()` below.
        // Bead `sts2-rs-3ebox`: C# `TimesUp.cs` OnUpgrade
        // `AddKeyword(Retain)`; spec EXHAUST → upgraded = EXHAUST | RETAIN.
        CardId::TimesUp => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(CardKeywords::EXHAUST.with(CardKeywords::RETAIN)),
        }),
        CardId::ToricToughness => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(7),
            keywords: None,
        }),
        // Bead `sts2-rs-3ebox`: C# `Transfigure.cs` OnUpgrade
        // `RemoveKeyword(Exhaust)`; spec EXHAUST → upgraded = NONE.
        // (OnPlay +1 max-energy rider is unrelated to upgrade and lives
        // in the handler.)
        CardId::Transfigure => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(CardKeywords::NONE),
        }),
        // Bead `sts2-rs-giozp`: C# `Tyranny.cs` OnUpgrade
        // `AddKeyword(Innate)`; spec NONE → INNATE.
        // Bead `sts2-rs-t8b3k`: C# `Undeath.cs` block 7 → 9.
        CardId::Undeath => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(9),
            keywords: None,
        }),
        // Bead `sts2-rs-tk230`: C# `Veilpiercer.cs` damage 10 → 13.
        CardId::Veilpiercer => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(13),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-giozp`: C# `VoidForm.cs` OnUpgrade
        // `RemoveKeyword(Ethereal)`; spec ETHEREAL → NONE.
        // Bead `sts2-rs-z13x4`: C# `Volley.cs` damage 10 → 14.
        CardId::Whistle => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(44),
            base_block: None,
            keywords: None,
        }),
        // Bead `sts2-rs-af3h4`: C# `WhiteNoise.cs` OnUpgrade
        // `EnergyCost.UpgradeBy(-1)` 1 → 0.
        CardId::Wish => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(CardKeywords::EXHAUST.with(CardKeywords::RETAIN)),
        }),
        // Bead `sts2-rs-3ebox`: C# `Wisp.cs` OnUpgrade
        // `AddKeyword(Retain)`; spec EXHAUST → upgraded = EXHAUST | RETAIN.
        CardId::Wisp => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(CardKeywords::EXHAUST.with(CardKeywords::RETAIN)),
        }),
        CardId::StrikeIronclad => crate::cards::ironclad::strike_ironclad::upgrade(),
        CardId::DefendIronclad => crate::cards::ironclad::defend_ironclad::upgrade(),
        CardId::Bash => crate::cards::ironclad::bash::upgrade(),
        CardId::Anger => crate::cards::ironclad::anger::upgrade(),
        CardId::BloodWall => crate::cards::ironclad::blood_wall::upgrade(),
        CardId::BodySlam => crate::cards::ironclad::body_slam::upgrade(),
        CardId::Breakthrough => crate::cards::ironclad::breakthrough::upgrade(),
        CardId::Cinder => crate::cards::ironclad::cinder::upgrade(),
        CardId::Havoc => crate::cards::ironclad::havoc::upgrade(),
        CardId::Headbutt => crate::cards::ironclad::headbutt::upgrade(),
        CardId::IronWave => crate::cards::ironclad::iron_wave::upgrade(),
        CardId::MoltenFist => crate::cards::ironclad::molten_fist::upgrade(),
        CardId::PommelStrike => crate::cards::ironclad::pommel_strike::upgrade(),
        CardId::SetupStrikeCard => crate::cards::ironclad::setup_strike::upgrade(),
        CardId::ShrugItOff => crate::cards::ironclad::shrug_it_off::upgrade(),
        CardId::Thunderclap => crate::cards::ironclad::thunderclap::upgrade(),
        CardId::TrueGrit => crate::cards::ironclad::true_grit::upgrade(),
        CardId::TwinStrike => crate::cards::ironclad::twin_strike::upgrade(),
        CardId::Bludgeon => crate::cards::ironclad::bludgeon::upgrade(),
        CardId::DemonicShield => crate::cards::ironclad::demonic_shield::upgrade(),
        CardId::Dismantle => crate::cards::ironclad::dismantle::upgrade(),
        CardId::Dominate => crate::cards::ironclad::dominate::upgrade(),
        CardId::EvilEye => crate::cards::ironclad::evil_eye::upgrade(),
        CardId::ExpectAFight => crate::cards::ironclad::expect_a_fight::upgrade(),
        CardId::FightMe => crate::cards::ironclad::fight_me::upgrade(),
        CardId::FlameBarrierCard => crate::cards::ironclad::flame_barrier::upgrade(),
        CardId::Hemokinesis => crate::cards::ironclad::hemokinesis::upgrade(),
        CardId::HowlFromBeyond => crate::cards::ironclad::howl_from_beyond::upgrade(),
        CardId::InfernalBlade => crate::cards::ironclad::infernal_blade::upgrade(),
        CardId::JugglingCard => crate::cards::ironclad::juggling::upgrade(),
        CardId::Pillage => crate::cards::ironclad::pillage::upgrade(),
        CardId::SecondWind => crate::cards::ironclad::second_wind::upgrade(),
        CardId::Spite => crate::cards::ironclad::spite::upgrade(),
        CardId::StampedeCard => crate::cards::ironclad::stampede::upgrade(),
        CardId::Stomp => crate::cards::ironclad::stomp::upgrade(),
        CardId::Taunt => crate::cards::ironclad::taunt::upgrade(),
        CardId::Unrelenting => crate::cards::ironclad::unrelenting::upgrade(),
        CardId::Whirlwind => crate::cards::ironclad::whirlwind::upgrade(),
        CardId::AggressionCard => crate::cards::ironclad::aggression::upgrade(),
        CardId::BarricadeCard => crate::cards::ironclad::barricade::upgrade(),
        CardId::ColossusCard => crate::cards::ironclad::colossus::upgrade(),
        CardId::Conflagration => crate::cards::ironclad::conflagration::upgrade(),
        CardId::DarkEmbraceCard => crate::cards::ironclad::dark_embrace::upgrade(),
        CardId::Feed => crate::cards::ironclad::feed::upgrade(),
        CardId::FiendFire => crate::cards::ironclad::fiend_fire::upgrade(),
        CardId::HellraiserCard => crate::cards::ironclad::hellraiser::upgrade(),
        CardId::Impervious => crate::cards::ironclad::impervious::upgrade(),
        CardId::Mangle => crate::cards::ironclad::mangle::upgrade(),
        CardId::PactsEnd => crate::cards::ironclad::pacts_end::upgrade(),
        CardId::Stoke => crate::cards::ironclad::stoke::upgrade(),
        CardId::TankCard => crate::cards::ironclad::tank::upgrade(),
        CardId::TearAsunder => crate::cards::ironclad::tear_asunder::upgrade(),
        CardId::Thrash => crate::cards::ironclad::thrash::upgrade(),
        CardId::Unmovable => crate::cards::ironclad::unmovable::upgrade(),
        CardId::Break => crate::cards::ironclad::r#break::upgrade(),
        CardId::CorruptionCard => crate::cards::ironclad::corruption::upgrade(),
        CardId::StrikeSilent => crate::cards::silent::strike_silent::upgrade(),
        CardId::DefendSilent => crate::cards::silent::defend_silent::upgrade(),
        CardId::Neutralize => crate::cards::silent::neutralize::upgrade(),
        CardId::Survivor => crate::cards::silent::survivor::upgrade(),
        CardId::Backflip => crate::cards::silent::backflip::upgrade(),
        CardId::DaggerSpray => crate::cards::silent::dagger_spray::upgrade(),
        CardId::DaggerThrow => crate::cards::silent::dagger_throw::upgrade(),
        CardId::Deflect => crate::cards::silent::deflect::upgrade(),
        CardId::DodgeAndRoll => crate::cards::silent::dodge_and_roll::upgrade(),
        CardId::FlickFlack => crate::cards::silent::flick_flack::upgrade(),
        CardId::LeadingStrike => crate::cards::silent::leading_strike::upgrade(),
        CardId::PoisonedStab => crate::cards::silent::poisoned_stab::upgrade(),
        CardId::Slice => crate::cards::silent::slice::upgrade(),
        CardId::SuckerPunch => crate::cards::silent::sucker_punch::upgrade(),
        CardId::Untouchable => crate::cards::silent::untouchable::upgrade(),
        CardId::Backstab => crate::cards::silent::backstab::upgrade(),
        CardId::BlurCard => crate::cards::silent::blur::upgrade(),
        CardId::CalculatedGamble => crate::cards::silent::calculated_gamble::upgrade(),
        CardId::Dash => crate::cards::silent::dash::upgrade(),
        CardId::EscapePlan => crate::cards::silent::escape_plan::upgrade(),
        CardId::Finisher => crate::cards::silent::finisher::upgrade(),
        CardId::Flanking => crate::cards::silent::flanking::upgrade(),
        CardId::Flechettes => crate::cards::silent::flechettes::upgrade(),
        CardId::Scare => crate::cards::silent::scare::upgrade(),
        CardId::HandTrick => crate::cards::silent::hand_trick::upgrade(),
        CardId::InfiniteBladesCard => crate::cards::silent::infinite_blades::upgrade(),
        CardId::LegSweep => crate::cards::silent::leg_sweep::upgrade(),
        CardId::MementoMori => crate::cards::silent::memento_mori::upgrade(),
        CardId::Mirage => crate::cards::silent::mirage::upgrade(),
        CardId::Pinpoint => crate::cards::silent::pinpoint::upgrade(),
        CardId::Pounce => crate::cards::silent::pounce::upgrade(),
        CardId::PreciseCut => crate::cards::silent::precise_cut::upgrade(),
        CardId::Predator => crate::cards::silent::predator::upgrade(),
        CardId::Skewer => crate::cards::silent::skewer::upgrade(),
        CardId::SpeedsterCard => crate::cards::silent::speedster::upgrade(),
        CardId::Strangle => crate::cards::silent::strangle::upgrade(),
        CardId::AfterimageCard => crate::cards::silent::afterimage::upgrade(),
        CardId::Assassinate => crate::cards::silent::assassinate::upgrade(),
        CardId::BulletTime => crate::cards::silent::bullet_time::upgrade(),
        CardId::EchoingSlash => crate::cards::silent::echoing_slash::upgrade(),
        CardId::GrandFinale => crate::cards::silent::grand_finale::upgrade(),
        CardId::MasterPlanner => crate::cards::silent::master_planner::upgrade(),
        CardId::Murder => crate::cards::silent::murder::upgrade(),
        CardId::Nightmare => crate::cards::silent::nightmare::upgrade(),
        CardId::ShadowStep => crate::cards::silent::shadow_step::upgrade(),
        CardId::Shadowmeld => crate::cards::silent::shadowmeld::upgrade(),
        CardId::TheHunt => crate::cards::silent::the_hunt::upgrade(),
        CardId::ToolsOfTheTrade => crate::cards::silent::tools_of_the_trade::upgrade(),
        CardId::Tracking => crate::cards::silent::tracking::upgrade(),
        CardId::Suppress => crate::cards::silent::suppress::upgrade(),
        CardId::StrikeDefect => crate::cards::defect::strike_defect::upgrade(),
        CardId::DefendDefect => crate::cards::defect::defend_defect::upgrade(),
        CardId::Zap => crate::cards::defect::zap::upgrade(),
        CardId::Dualcast => crate::cards::defect::dualcast::upgrade(),
        CardId::BallLightning => crate::cards::defect::ball_lightning::upgrade(),
        CardId::Barrage => crate::cards::defect::barrage::upgrade(),
        CardId::BeamCell => crate::cards::defect::beam_cell::upgrade(),
        CardId::BoostAway => crate::cards::defect::boost_away::upgrade(),
        CardId::ChargeBattery => crate::cards::defect::charge_battery::upgrade(),
        CardId::Claw => crate::cards::defect::claw::upgrade(),
        CardId::ColdSnap => crate::cards::defect::cold_snap::upgrade(),
        CardId::CompileDriver => crate::cards::defect::compile_driver::upgrade(),
        CardId::FocusedStrikeCard => crate::cards::defect::focused_strike::upgrade(),
        CardId::GoForTheEyes => crate::cards::defect::go_for_the_eyes::upgrade(),
        CardId::GunkUp => crate::cards::defect::gunk_up::upgrade(),
        CardId::Hologram => crate::cards::defect::hologram::upgrade(),
        CardId::Hotfix => crate::cards::defect::hotfix::upgrade(),
        CardId::Leap => crate::cards::defect::leap::upgrade(),
        CardId::LightningRod => crate::cards::defect::lightning_rod::upgrade(),
        CardId::MomentumStrike => crate::cards::defect::momentum_strike::upgrade(),
        CardId::SweepingBeam => crate::cards::defect::sweeping_beam::upgrade(),
        CardId::Uproar => crate::cards::defect::uproar::upgrade(),
        CardId::BootSequence => crate::cards::defect::boot_sequence::upgrade(),
        CardId::Chill => crate::cards::defect::chill::upgrade(),
        CardId::Compact => crate::cards::defect::compact::upgrade(),
        CardId::DoubleEnergy => crate::cards::defect::double_energy::upgrade(),
        CardId::Feral => crate::cards::defect::feral::upgrade(),
        CardId::FightThrough => crate::cards::defect::fight_through::upgrade(),
        CardId::Ftl => crate::cards::defect::ftl::upgrade(),
        CardId::Fusion => crate::cards::defect::fusion::upgrade(),
        CardId::Glacier => crate::cards::defect::glacier::upgrade(),
        CardId::Glasswork => crate::cards::defect::glasswork::upgrade(),
        CardId::NullCard => crate::cards::defect::null::upgrade(),
        CardId::Refract => crate::cards::defect::refract::upgrade(),
        CardId::RocketPunch => crate::cards::defect::rocket_punch::upgrade(),
        CardId::Scrape => crate::cards::defect::scrape::upgrade(),
        CardId::ShadowShield => crate::cards::defect::shadow_shield::upgrade(),
        CardId::Subroutine => crate::cards::defect::subroutine::upgrade(),
        CardId::Sunder => crate::cards::defect::sunder::upgrade(),
        CardId::Synchronize => crate::cards::defect::synchronize::upgrade(),
        CardId::Synthesis => crate::cards::defect::synthesis::upgrade(),
        CardId::TeslaCoil => crate::cards::defect::tesla_coil::upgrade(),
        CardId::WhiteNoise => crate::cards::defect::white_noise::upgrade(),
        CardId::AdaptiveStrike => crate::cards::defect::adaptive_strike::upgrade(),
        CardId::AllForOne => crate::cards::defect::all_for_one::upgrade(),
        CardId::CreativeAiCard => crate::cards::defect::creative_ai::upgrade(),
        CardId::EchoFormCard => crate::cards::defect::echo_form::upgrade(),
        CardId::FlakCannon => crate::cards::defect::flak_cannon::upgrade(),
        CardId::HelixDrill => crate::cards::defect::helix_drill::upgrade(),
        CardId::Hyperbeam => crate::cards::defect::hyperbeam::upgrade(),
        CardId::IceLance => crate::cards::defect::ice_lance::upgrade(),
        CardId::Ignition => crate::cards::defect::ignition::upgrade(),
        CardId::MachineLearningCard => crate::cards::defect::machine_learning::upgrade(),
        CardId::MeteorStrike => crate::cards::defect::meteor_strike::upgrade(),
        CardId::Rainbow => crate::cards::defect::rainbow::upgrade(),
        CardId::Shatter => crate::cards::defect::shatter::upgrade(),
        CardId::SignalBoost => crate::cards::defect::signal_boost::upgrade(),
        CardId::TrashToTreasure => crate::cards::defect::trash_to_treasure::upgrade(),
        CardId::Voltaic => crate::cards::defect::voltaic::upgrade(),
        CardId::Quadcast => crate::cards::defect::quadcast::upgrade(),
        CardId::Arsenal => crate::cards::regent::arsenal::upgrade(),
        CardId::AstralPulse => crate::cards::regent::astral_pulse::upgrade(),
        CardId::BeatIntoShape => crate::cards::regent::beat_into_shape::upgrade(),
        CardId::Begone => crate::cards::regent::begone::upgrade(),
        CardId::BigBang => crate::cards::regent::big_bang::upgrade(),
        CardId::Bombardment => crate::cards::regent::bombardment::upgrade(),
        CardId::Bulwark => crate::cards::regent::bulwark::upgrade(),
        CardId::CelestialMight => crate::cards::regent::celestial_might::upgrade(),
        CardId::CloakOfStars => crate::cards::regent::cloak_of_stars::upgrade(),
        CardId::CollisionCourse => crate::cards::regent::collision_course::upgrade(),
        CardId::Comet => crate::cards::regent::comet::upgrade(),
        CardId::CosmicIndifference => crate::cards::regent::cosmic_indifference::upgrade(),
        CardId::CrashLanding => crate::cards::regent::crash_landing::upgrade(),
        CardId::CrushUnder => crate::cards::regent::crush_under::upgrade(),
        CardId::DefendRegent => crate::cards::regent::defend_regent::upgrade(),
        CardId::Devastate => crate::cards::regent::devastate::upgrade(),
        CardId::DyingStar => crate::cards::regent::dying_star::upgrade(),
        CardId::FallingStar => crate::cards::regent::falling_star::upgrade(),
        CardId::GammaBlast => crate::cards::regent::gamma_blast::upgrade(),
        CardId::GatherLight => crate::cards::regent::gather_light::upgrade(),
        CardId::Glitterstream => crate::cards::regent::glitterstream::upgrade(),
        CardId::GuidingStar => crate::cards::regent::guiding_star::upgrade(),
        CardId::HammerTime => crate::cards::regent::hammer_time::upgrade(),
        CardId::HeavenlyDrill => crate::cards::regent::heavenly_drill::upgrade(),
        CardId::Hegemony => crate::cards::regent::hegemony::upgrade(),
        CardId::HeirloomHammer => crate::cards::regent::heirloom_hammer::upgrade(),
        CardId::IAmInvincible => crate::cards::regent::i_am_invincible::upgrade(),
        CardId::KinglyKick => crate::cards::regent::kingly_kick::upgrade(),
        CardId::KinglyPunch => crate::cards::regent::kingly_punch::upgrade(),
        CardId::KnockoutBlow => crate::cards::regent::knockout_blow::upgrade(),
        CardId::KnowThyPlace => crate::cards::regent::know_thy_place::upgrade(),
        CardId::LunarBlast => crate::cards::regent::lunar_blast::upgrade(),
        CardId::MakeItSo => crate::cards::regent::make_it_so::upgrade(),
        CardId::ManifestAuthority => crate::cards::regent::manifest_authority::upgrade(),
        CardId::MeteorShower => crate::cards::regent::meteor_shower::upgrade(),
        CardId::MonarchsGazeCard => crate::cards::regent::monarchs_gaze_card::upgrade(),
        CardId::MonologueCard => crate::cards::regent::monologue_card::upgrade(),
        CardId::Orbit => crate::cards::regent::orbit::upgrade(),
        CardId::ParticleWall => crate::cards::regent::particle_wall::upgrade(),
        CardId::Patter => crate::cards::regent::patter::upgrade(),
        CardId::PhotonCut => crate::cards::regent::photon_cut::upgrade(),
        CardId::PillarOfCreation => crate::cards::regent::pillar_of_creation::upgrade(),
        CardId::Radiate => crate::cards::regent::radiate::upgrade(),
        CardId::ReflectCard => crate::cards::regent::reflect_card::upgrade(),
        CardId::RoyalGamble => crate::cards::regent::royal_gamble::upgrade(),
        CardId::SevenStars => crate::cards::regent::seven_stars::upgrade(),
        CardId::ShiningStrike => crate::cards::regent::shining_strike::upgrade(),
        CardId::SolarStrike => crate::cards::regent::solar_strike::upgrade(),
        CardId::SpectrumShift => crate::cards::regent::spectrum_shift::upgrade(),
        CardId::Stardust => crate::cards::regent::stardust::upgrade(),
        CardId::StrikeRegent => crate::cards::regent::strike_regent::upgrade(),
        CardId::SwordSage => crate::cards::regent::sword_sage::upgrade(),
        CardId::TheSealedThrone => crate::cards::regent::the_sealed_throne::upgrade(),
        CardId::TyrannyCard => crate::cards::regent::tyranny_card::upgrade(),
        CardId::VoidForm => crate::cards::regent::void_form::upgrade(),
        CardId::WroughtInWar => crate::cards::regent::wrought_in_war::upgrade(),
        CardId::Discovery => crate::cards::colorless::discovery::upgrade(),
        CardId::DramaticEntrance => crate::cards::colorless::dramatic_entrance::upgrade(),
        CardId::Finesse => crate::cards::colorless::finesse::upgrade(),
        CardId::FlashOfSteel => crate::cards::colorless::flash_of_steel::upgrade(),
        CardId::Omnislice => crate::cards::colorless::omnislice::upgrade(),
        CardId::SecretTechnique => crate::cards::colorless::secret_technique::upgrade(),
        CardId::SecretWeapon => crate::cards::colorless::secret_weapon::upgrade(),
        CardId::ThinkingAhead => crate::cards::colorless::thinking_ahead::upgrade(),
        CardId::Alchemize => crate::cards::colorless::alchemize::upgrade(),
        CardId::HandOfGreed => crate::cards::colorless::hand_of_greed::upgrade(),
        CardId::Fisticuffs => crate::cards::colorless::fisticuffs::upgrade(),
        CardId::PanicButton => crate::cards::colorless::panic_button::upgrade(),
        CardId::UltimateStrike => crate::cards::colorless::ultimate_strike::upgrade(),
        CardId::UltimateDefend => crate::cards::colorless::ultimate_defend::upgrade(),
        CardId::Automation => crate::cards::colorless::automation::upgrade(),
        CardId::Equilibrium => crate::cards::colorless::equilibrium::upgrade(),
        CardId::InterceptCard => crate::cards::colorless::intercept_card::upgrade(),
        CardId::Lift => crate::cards::colorless::lift::upgrade(),
        CardId::MindBlast => crate::cards::colorless::mind_blast::upgrade(),
        CardId::Production => crate::cards::colorless::production::upgrade(),
        CardId::Prolong => crate::cards::colorless::prolong::upgrade(),
        CardId::SeekerStrike => crate::cards::colorless::seeker_strike::upgrade(),
        CardId::Stratagem => crate::cards::colorless::stratagem::upgrade(),
        CardId::TagTeam => crate::cards::colorless::tag_team::upgrade(),
        CardId::ThrummingHatchet => crate::cards::colorless::thrumming_hatchet::upgrade(),
        CardId::Volley => crate::cards::colorless::volley::upgrade(),
        CardId::Anointed => crate::cards::colorless::anointed::upgrade(),
        CardId::BeaconOfHope => crate::cards::colorless::beacon_of_hope::upgrade(),
        CardId::Bolas => crate::cards::colorless::bolas::upgrade(),
        CardId::CalamityCard => crate::cards::colorless::calamity_card::upgrade(),
        CardId::Entropy => crate::cards::colorless::entropy::upgrade(),
        CardId::GoldAxe => crate::cards::colorless::gold_axe::upgrade(),
        CardId::Jackpot => crate::cards::colorless::jackpot::upgrade(),
        CardId::Knockdown => crate::cards::colorless::knockdown::upgrade(),
        CardId::MayhemCard => crate::cards::colorless::mayhem_card::upgrade(),
        CardId::Mimic => crate::cards::colorless::mimic::upgrade(),
        CardId::NostalgiaCard => crate::cards::colorless::nostalgia_card::upgrade(),
        CardId::Rally => crate::cards::colorless::rally::upgrade(),
        CardId::Salvo => crate::cards::colorless::salvo::upgrade(),
        CardId::Scrawl => crate::cards::colorless::scrawl::upgrade(),
        CardId::TheGambit => crate::cards::colorless::the_gambit::upgrade(),
        // ── v0.113.0-only event cards. Absolute upgraded values.
        // `DingyDagger.cs`: first `OnUpgrade` adds 3 + level(1) = 4 → 11.
        CardId::DingyDagger => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(11),
            base_block: None,
            keywords: None,
        }),
        // `MightyWeapon.cs`: `Damage.UpgradeValueBy(5)` 25 → 30.
        CardId::MightyWeapon => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(30),
            base_block: None,
            keywords: None,
        }),
        // `MyShinies.cs`: `Gold.UpgradeValueBy(-10)` (20 → 10 gold per hit);
        // no spec field moves — the handler reads the upgraded flag.
        CardId::MyShinies => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // `MightyShield.cs`: `Block.UpgradeValueBy(5)` 20 → 25.
        CardId::MightyShield => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(25),
            keywords: None,
        }),
        // ── v0.111.0-only cards (GH #793). The ids exist in no v0.107.1
        // state, so these rows cannot move a v0.107.1 upgrade.
        // `Abundance.cs` / `Tutor.cs`: `EnergyCost.UpgradeBy(-1)` 1 → 0.
        CardId::Abundance | CardId::Tutor => Some(CardUpgradeDelta {
            cost: Some(0),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // `BladeSymphony.cs`: `EnergyCost.UpgradeBy(-1)` 2 → 1.
        CardId::BladeSymphony => Some(CardUpgradeDelta {
            cost: Some(1),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // `Constellation.cs`: `Block.UpgradeValueBy(3)` 9 → 12.
        CardId::Constellation => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(12),
            keywords: None,
        }),
        // `Midnight.cs`: `Damage.UpgradeValueBy(12)` 60 → 72.
        CardId::Midnight => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(72),
            base_block: None,
            keywords: None,
        }),
        // `Outrage.cs`: `Damage.UpgradeValueBy(4)` 9 → 13.
        CardId::Outrage => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(13),
            base_block: None,
            keywords: None,
        }),
        // `Soulbound.cs`: `AddKeyword(Innate)`; spec NONE → INNATE.
        CardId::SoulboundCard => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(CardKeywords::INNATE),
        }),
        // `Underworld.cs`: `RemoveKeyword(Exhaust)`; spec EXHAUST → NONE.
        CardId::UnderworldCard => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(CardKeywords::NONE),
        }),
        // v0.113.0 Fool cards: `crate::cards::fool::specs::upgrade_delta`.
        _ => super::fool::specs::upgrade_delta(card_id),
    }
}
