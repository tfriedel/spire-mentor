//! Affliction registry — bead `sts2-rs-d19` (P7-8).
//!
//! Classifies every [`PowerKind`] variant by polarity (Buff / Debuff /
//! Mixed / Neutral) and duration semantics (`TurnEnd` / Permanent /
//! `UntilCleared` / Stack). Used by:
//!
//! * The affliction-cleanup pass at `player_turn_end` (decrement
//!   `TurnEnd` debuffs).
//! * Card / relic handlers that read polarity (e.g. Artifact only
//!   blocks Debuffs).
//! * Parity tests that assert every variant has an entry — catches
//!   future variant additions that forget to classify.
//!
//! `classify` is total over [`PowerKind`]; unrecognized long-tail
//! variants get [`AfflictionClass::UNKNOWN`] (treated as Neutral by
//! cleanup). The classification floor is enforced by the test
//! `classification_coverage_floor` below + by
//! `tests/affliction_invariants.rs`.

use crate::PowerKind;

/// Polarity of a power instance from the player's perspective.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AfflictionKind {
    /// Positive — Strength, Dexterity, Vigor, Plating, …
    Buff,
    /// Negative — Vulnerable, Weak, Frail, Poison, …
    Debuff,
    /// Either-or depending on owner.
    Mixed,
    /// Bookkeeping / state-machine markers with no polarity.
    Neutral,
    /// Long-tail variant not yet classified.
    Unknown,
}

/// Duration semantics — how the `amount` field decays each turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AfflictionDuration {
    /// Decrement by 1 at the owner's turn end.
    TurnEnd,
    /// `amount` is a stack count consumed by trigger events.
    Stack,
    /// Persists for the whole combat (Strength, Ritual, …).
    Permanent,
    /// Persists until a cleansing effect fires (Artifact, …).
    UntilCleared,
}

/// Full classification record for one [`PowerKind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AfflictionClass {
    pub kind: AfflictionKind,
    pub duration: AfflictionDuration,
}

impl AfflictionClass {
    pub const UNKNOWN: Self = Self {
        kind: AfflictionKind::Unknown,
        duration: AfflictionDuration::Permanent,
    };

    const fn buff_perm() -> Self {
        Self {
            kind: AfflictionKind::Buff,
            duration: AfflictionDuration::Permanent,
        }
    }
    const fn buff_turn() -> Self {
        Self {
            kind: AfflictionKind::Buff,
            duration: AfflictionDuration::TurnEnd,
        }
    }
    const fn buff_stack() -> Self {
        Self {
            kind: AfflictionKind::Buff,
            duration: AfflictionDuration::Stack,
        }
    }
    const fn buff_clear() -> Self {
        Self {
            kind: AfflictionKind::Buff,
            duration: AfflictionDuration::UntilCleared,
        }
    }
    const fn debuff_turn() -> Self {
        Self {
            kind: AfflictionKind::Debuff,
            duration: AfflictionDuration::TurnEnd,
        }
    }
    const fn debuff_stack() -> Self {
        Self {
            kind: AfflictionKind::Debuff,
            duration: AfflictionDuration::Stack,
        }
    }
    const fn debuff_perm() -> Self {
        Self {
            kind: AfflictionKind::Debuff,
            duration: AfflictionDuration::Permanent,
        }
    }
    const fn neutral() -> Self {
        Self {
            kind: AfflictionKind::Neutral,
            duration: AfflictionDuration::Permanent,
        }
    }

    /// `true` when this class is a debuff (affliction).
    #[must_use]
    pub const fn is_debuff(self) -> bool {
        matches!(self.kind, AfflictionKind::Debuff)
    }
    /// `true` when this class is a buff.
    #[must_use]
    pub const fn is_buff(self) -> bool {
        matches!(self.kind, AfflictionKind::Buff)
    }
}

/// Classify a single [`PowerKind`]. Total over the enum.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn classify(kind: PowerKind) -> AfflictionClass {
    match kind {
        // ---------- Core pipeline debuffs (turn-decay) ----------
        PowerKind::Vulnerable
        | PowerKind::Weak
        | PowerKind::Frail
        | PowerKind::NoBlock
        | PowerKind::Shrink
        | PowerKind::Knockdown
        | PowerKind::Hang
        | PowerKind::Shadowmeld
        | PowerKind::Entangled
        | PowerKind::Confused
        | PowerKind::Fasten
        | PowerKind::NoDraw => AfflictionClass::debuff_turn(),

        // ---------- Stack-style debuffs ----------
        PowerKind::Poison
        | PowerKind::NoxiousFumes
        | PowerKind::BiasedCognition
        | PowerKind::CorrosiveWave => AfflictionClass::debuff_stack(),

        // ---------- Permanent / stat-style debuffs ----------
        // Slow's `amount` is a stack count that does NOT decay at turn end
        // (C# `SlowPower` never decrements Amount). Its scaling lives in the
        // per-instance `counter` (SlowAmount), bumped per card played and
        // reset at the owner's turn start — see the `Slow` arms in `hooks`.
        //
        // Doom: C# `DoomPower.Type == PowerType.Debuff` (DoomPower.cs:22).
        // Its `Amount` is a doom threshold that never turn-decays
        // (`DoomPower.BeforeTurnEnd` doom-kills but never decrements Amount),
        // so it classifies as a permanent debuff. Consumers that read
        // `is_debuff()` — Misery / Rend (count debuffs on target),
        // UnsettlingLamp + SleightOfFlesh (player-applied-debuff-to-enemy
        // gates) — now correctly include Doom. Bead `sts2-rs-30xk8`.
        PowerKind::TheBomb
        | PowerKind::MagicBomb
        | PowerKind::Surrounded
        | PowerKind::Slow
        | PowerKind::Doom => AfflictionClass::debuff_perm(),

        // ---------- C# `PowerType.Debuff` audit (bead `sts2-rs-t88vp`) ----------
        // Variants whose C# `<X>Power.Type == PowerType.Debuff` (verified
        // against ../sts2-wiki/decompiled/v0.103.2/.../Powers/<X>Power.cs) but
        // that previously fell into the `_ => UNKNOWN` arm, so
        // `is_debuff()` under-fired for the Misery/Rend debuff-count and the
        // UnsettlingLamp/SleightOfFlesh/Artifact debuff gates.
        //
        // Polarity here is authoritative (from the C# `Type` override). The
        // `AfflictionDuration` field is currently DIAGNOSTIC-ONLY — no
        // production code reads `classify(...).duration`; each power's actual
        // turn-end decay (or persistence) is hand-coded in its `hooks.rs`
        // `before_turn_end`/`after_turn_end` arm (e.g. Constrict/Disintegration
        // persist as Counter stacks; Tangled/BorrowedTime self-remove at turn
        // end; the temp-strength-down family reverts at `AfterTurnEnd`). We
        // therefore default the inert duration to `debuff_perm` (asserts no
        // implied auto-decay) rather than guess per-power decay; consult the
        // hook for the real lifetime if `duration` ever becomes load-bearing.
        //
        // The 8 inherited debuffs (CrushUnder/DarkShackles/DyingStar/
        // EnfeeblingTouch/Mangle/MonarchsGazeStrengthDown/PiercingWail/
        // ShacklingPotion) derive from `TemporaryStrengthPower` with
        // `IsPositive => false`, so their amount-dependent `Type` is fixed to
        // Debuff. Amount-dependent powers that are positive-when-applied
        // (Coordinate/FeedingFrenzy/FlexPotion/ReptileTrinket/SetupStrike/
        // Anticipate/HelicalDart/SpeedPotion/FocusedStrike/Hotfix/Synchronize)
        // are NOT debuffs in practice and remain UNKNOWN (is_debuff stays false,
        // which is what their positive instances want).
        PowerKind::BorrowedTime
        | PowerKind::ChainsOfBinding
        | PowerKind::Conqueror
        | PowerKind::Constrict
        | PowerKind::Dampen
        | PowerKind::Debilitate
        | PowerKind::Demise
        | PowerKind::Disintegration
        | PowerKind::Flanking
        | PowerKind::Hex
        | PowerKind::Imbalanced
        | PowerKind::MindRot
        | PowerKind::NoEnergyGain
        | PowerKind::Oblivion
        | PowerKind::Plow
        | PowerKind::Ringing
        | PowerKind::Shriek
        // `SicEmPower : PowerModel` with `Type => Debuff` (SicEmPower.cs:16).
        // The Necrobinder SicEm card applies `PowerKind::SicEmPower` (the
        // variant every handler/listener uses); the vestigial `PowerKind::SicEm`
        // is never applied. Classifying `SicEmPower` routes
        // `apply_power_to_enemy(SicEmPower)` through the Artifact-aware debuff
        // path so an Artifact-warded enemy consumes a stack and negates the mark
        // (parity fuzz: SIC_EM seeds 136/306 — Rust applied SicEm past Artifact).
        | PowerKind::SicEmPower
        | PowerKind::Sloth
        | PowerKind::Smoggy
        | PowerKind::Strangle
        | PowerKind::TagTeam
        | PowerKind::Tangled
        | PowerKind::Tender
        | PowerKind::TheGambit
        | PowerKind::WasteAway
        | PowerKind::WraithForm
        | PowerKind::CrushUnder
        | PowerKind::DarkShackles
        | PowerKind::DyingStar
        | PowerKind::EnfeeblingTouch
        | PowerKind::Mangle
        // `ManglePower : TemporaryStrengthPower` with `IsPositive=false`, so
        // its `Type` is `Debuff` (`TemporaryStrengthPower.cs:18-28`) and it
        // is applied with a POSITIVE amount, so `GetTypeForAmount(+amt) ==
        // Debuff` (`PowerModel.cs:400`). Artifact therefore zeroes/consumes
        // it on application (`ArtifactPower.cs:22-23`). Classifying it here
        // routes `apply_power_to_enemy(ManglePower)` through the
        // Artifact-aware debuff path. Bead `sts2-rs-q6bix` (MechaKnight).
        | PowerKind::ManglePower
        | PowerKind::MonarchsGazeStrengthDown
        | PowerKind::PiercingWail
        | PowerKind::ShacklingPotion
        // v0.111.0 `HyperbeamFocusDownPower : TemporaryFocusPower` with
        // `IsPositive => false`, so `Type => Debuff` (decomp
        // HyperbeamFocusDownPower.cs:7, TemporaryFocusPower.cs:20-30).
        | PowerKind::HyperbeamFocusDown
        // v0.113.0 Fool: `DislikeYouPower` / `EncorePower` are `Type => Debuff`
        // (removed by their own turn-end hooks, not by decay).
        | PowerKind::DislikeYou
        | PowerKind::Encore => AfflictionClass::debuff_perm(),

        // ---------- Core pipeline buffs (permanent / stat) ----------
        PowerKind::Strength
        | PowerKind::Dexterity
        | PowerKind::Focus
        | PowerKind::Vigor
        | PowerKind::Intangible
        | PowerKind::Thorns
        | PowerKind::Plating
        | PowerKind::Buffer
        | PowerKind::DemonForm
        | PowerKind::Metallicize
        | PowerKind::Ritual
        | PowerKind::Barricade
        | PowerKind::Corruption
        | PowerKind::FeelNoPain
        | PowerKind::Juggernaut
        | PowerKind::Rupture
        | PowerKind::DarkEmbrace
        | PowerKind::DoubleDamage
        | PowerKind::FlameBarrier
        | PowerKind::Afterimage
        | PowerKind::EchoForm
        | PowerKind::HelloWorld
        | PowerKind::MachineLearning
        | PowerKind::Loop
        | PowerKind::Storm
        | PowerKind::Neurosurge
        | PowerKind::CrimsonMantle
        | PowerKind::Inferno => AfflictionClass::buff_perm(),

        // ---------- Turn-decay buffs ----------
        PowerKind::Blur
        | PowerKind::EnergyNextTurn
        | PowerKind::DrawCardsNextTurn
        | PowerKind::BlockNextTurn => AfflictionClass::buff_turn(),

        // ---------- Stack-style buffs ----------
        PowerKind::Accuracy
        | PowerKind::PhantomBlades
        | PowerKind::Lethality
        | PowerKind::Leadership => AfflictionClass::buff_stack(),

        // ---------- Until-cleared buffs ----------
        PowerKind::Artifact => AfflictionClass::buff_clear(),

        // ---------- Monster bookkeeping / phase markers ----------
        PowerKind::Minion
        | PowerKind::PainfulStabs
        | PowerKind::Enrage
        | PowerKind::CurlUp
        | PowerKind::Rampart
        | PowerKind::Thievery
        | PowerKind::BackAttackLeft
        | PowerKind::BackAttackRight
        | PowerKind::BattlewornDummyTimeLimit
        | PowerKind::Countdown
        | PowerKind::Aggression
        | PowerKind::Coolant
        | PowerKind::PrepTime
        | PowerKind::RollingBoulder
        | PowerKind::Sandpit
        | PowerKind::ShadowStep => AfflictionClass::neutral(),

        // ---------- Fallback: long-tail variants not yet classified ----------
        _ => AfflictionClass::UNKNOWN,
    }
}

/// `true` when the power-kind is a debuff (affliction).
#[must_use]
pub fn is_affliction(kind: PowerKind) -> bool {
    classify(kind).is_debuff()
}

/// The `AllowNegative` powers — C# `StrengthPower`/`DexterityPower`/
/// `FocusPower` (`Type => Buff`) and `ShriekPower`/`ShrinkPower`
/// (`Type => Debuff`), all declaring `AllowNegative => true`
/// (`MegaCrit.Sts2.Core.Models.Powers/*.cs`). At a negative amount every one
/// of these reads as a `Debuff` (a `Counter` + `AllowNegative` power hits C#
/// `GetTypeForAmount` rule 1; the Debuff-typed ones land on `Debuff` either
/// way). The Buff-typed three are the ones whose classification actually
/// *flips* with sign; Shriek/Shrink are listed so C# rule 2 (below) does not
/// mistakenly flip them to Buff.
const fn is_allow_negative(kind: PowerKind) -> bool {
    matches!(
        kind,
        PowerKind::Strength
            | PowerKind::Dexterity
            | PowerKind::Focus
            | PowerKind::Shriek
            | PowerKind::Shrink
    )
}

/// Polarity of a power *for its current signed amount*, mirroring C#
/// `PowerModel.GetTypeForAmount` (`PowerModel.cs:400`):
/// an `AllowNegative` power (Strength/Dexterity/Focus/Shriek/Shrink) reads as
/// `Debuff` while its amount is negative — even the Buff-typed attribute
/// powers; a non-`AllowNegative` `Debuff` power with a negative amount reads
/// as `Buff`. All other cases fall through to the static [`classify`].
///
/// Consumers that inspect *existing* powers on a creature by their live
/// polarity — C# `power.TypeForCurrentAmount == PowerType.Debuff` — must use
/// this rather than [`classify`], which is sign-agnostic. Today that's
/// Rend (count target debuffs) and Misery (snapshot target debuffs).
#[must_use]
pub fn type_for_amount(kind: PowerKind, amount: i32) -> AfflictionKind {
    if amount < 0 {
        // C# rule 1: AllowNegative power at a negative amount → Debuff.
        if is_allow_negative(kind) {
            return AfflictionKind::Debuff;
        }
        // C# rule 2: a !AllowNegative Debuff-typed power at a negative amount
        // reads as a Buff. Rare in practice (debuffs apply positive), but
        // mirrored for faithfulness.
        if classify(kind).is_debuff() {
            return AfflictionKind::Buff;
        }
    }
    classify(kind).kind
}

/// `true` when `kind` reads as a debuff for its current signed `amount`
/// (C# `TypeForCurrentAmount == PowerType.Debuff`). See [`type_for_amount`].
#[must_use]
pub fn is_debuff_for_amount(kind: PowerKind, amount: i32) -> bool {
    matches!(type_for_amount(kind, amount), AfflictionKind::Debuff)
}

/// `true` when `kind` is the Rust analogue of a C# `ITemporaryPower`
/// **debuff** — a `TemporaryStrengthPower` subclass with `IsPositive => false`.
///
/// C# `Rend.cs::ShouldCountPower`
/// (`../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/Rend.cs:44-48`)
/// counts a target power only when
/// `power.TypeForCurrentAmount == PowerType.Debuff && !(power is ITemporaryPower)`
/// — i.e. it EXCLUDES every `ITemporaryPower`. `ITemporaryPower`
/// (`MegaCrit.Sts2.Core.Models/ITemporaryPower.cs`) is implemented by
/// `TemporaryStrengthPower` / `TemporaryDexterityPower` / `TemporaryFocusPower`
/// and their subclasses. Of those, only the `TemporaryStrengthPower` subclasses
/// with `IsPositive => false` read as Debuffs (so they would otherwise pass
/// [`is_debuff_for_amount`] and be miscounted by Rend); the Focus/Dexterity
/// subclasses and the positive Strength subclasses
/// (`FlexPotion`/`FeedingFrenzy`/`Coordinate`/`ReptileTrinket`/`SetupStrike`,
/// `Anticipate`/`HelicalDart`/`SpeedPotion`/`FocusedStrike`/`Hotfix`/`Synchronize`)
/// are buffs and are never counted by Rend's debuff filter anyway.
///
/// The 8 debuff `TemporaryStrengthPower` subclasses (C# class → Rust variant):
/// `CrushUnderPower` → [`PowerKind::CrushUnder`], `DarkShacklesPower` →
/// [`PowerKind::DarkShackles`], `DyingStarPower` → [`PowerKind::DyingStar`],
/// `EnfeeblingTouchPower` → [`PowerKind::EnfeeblingTouch`], `ManglePower` →
/// [`PowerKind::ManglePower`] (and the legacy [`PowerKind::Mangle`] alias),
/// `MonarchsGazeStrengthDownPower` → [`PowerKind::MonarchsGazeStrengthDown`],
/// `PiercingWailPower` → [`PowerKind::PiercingWail`], `ShacklingPotionPower` →
/// [`PowerKind::ShacklingPotion`].
///
/// Used by Rend's debuff-count filter to mirror the `!(power is ITemporaryPower)`
/// exclusion. (Note Misery snapshots target debuffs too, but C# Misery does NOT
/// apply this exclusion, so it must keep using [`is_debuff_for_amount`] alone.)
#[must_use]
pub fn is_temporary_power_debuff(kind: PowerKind) -> bool {
    matches!(
        kind,
        PowerKind::CrushUnder
            | PowerKind::DarkShackles
            | PowerKind::DyingStar
            | PowerKind::EnfeeblingTouch
            | PowerKind::Mangle
            | PowerKind::ManglePower
            | PowerKind::MonarchsGazeStrengthDown
            | PowerKind::PiercingWail
            | PowerKind::ShacklingPotion
    )
}

/// `true` when `kind` is the Rust analogue of C# `TemporaryStrengthPower`
/// or ANY of its subclasses — the set an `owner.GetPowerInstances
/// <TemporaryStrengthPower>()` walk matches (an `is`-typed walk, so every
/// subclass qualifies regardless of `IsPositive`).
///
/// Consumer: `PlowPower.AfterDamageReceived`
/// (`MegaCrit.Sts2.Core.Models.Powers/PlowPower.cs:35-39`) removes every
/// `TemporaryStrengthPower` instance from the CeremonialBeast BEFORE the
/// plain `PowerCmd.Remove<StrengthPower>` — load-bearing, because a
/// surviving instance's `TemporaryStrengthPower.AfterSideTurnEnd`
/// (`TemporaryStrengthPower.cs:167-176`) would refund `-Sign * Amount`
/// Strength at the owner's turn end, resurrecting Strength the stun just
/// wiped.
///
/// The 14 members (C# class → Rust variant): the base
/// `TemporaryStrengthPower` → [`PowerKind::TemporaryStrength`], plus
/// `Coordinate` / `CrushUnder` / `DarkShackles` / `DyingStar` /
/// `EnfeeblingTouch` / `FeedingFrenzy` / `FlexPotion` / `Mangle`(+
/// legacy [`PowerKind::ManglePower`] alias) / `MonarchsGazeStrengthDown` /
/// `PiercingWail` / `ReptileTrinket` / `SetupStrike` / `ShacklingPotion`.
/// (`TemporaryDexterityPower` / `TemporaryFocusPower` subclasses are NOT
/// `TemporaryStrengthPower` and are excluded, as in C#.)
#[must_use]
pub fn is_temporary_strength_family(kind: PowerKind) -> bool {
    matches!(
        kind,
        PowerKind::TemporaryStrength
            | PowerKind::Coordinate
            | PowerKind::CrushUnder
            | PowerKind::DarkShackles
            | PowerKind::DyingStar
            | PowerKind::EnfeeblingTouch
            | PowerKind::FeedingFrenzy
            | PowerKind::FlexPotion
            | PowerKind::Mangle
            | PowerKind::ManglePower
            | PowerKind::MonarchsGazeStrengthDown
            | PowerKind::PiercingWail
            | PowerKind::ReptileTrinket
            | PowerKind::SetupStrike
            | PowerKind::ShacklingPotion
    )
}

/// Number of variants in each classification bucket. Used by the
/// invariants test + audit report.
#[must_use]
pub fn coverage_counts() -> CoverageCounts {
    let mut c = CoverageCounts::default();
    for &k in PowerKind::all() {
        match classify(k).kind {
            AfflictionKind::Buff => c.buffs += 1,
            AfflictionKind::Debuff => c.debuffs += 1,
            AfflictionKind::Mixed => c.mixed += 1,
            AfflictionKind::Neutral => c.neutrals += 1,
            AfflictionKind::Unknown => c.unknown += 1,
        }
    }
    c.total = PowerKind::all().len();
    c
}

/// Coverage record returned by [`coverage_counts`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CoverageCounts {
    pub buffs: usize,
    pub debuffs: usize,
    pub mixed: usize,
    pub neutrals: usize,
    pub unknown: usize,
    pub total: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_is_total_over_power_kind() {
        for &k in PowerKind::all() {
            let _ = classify(k);
        }
    }

    #[test]
    fn known_afflictions_are_debuffs() {
        for k in [
            PowerKind::Vulnerable,
            PowerKind::Weak,
            PowerKind::Frail,
            PowerKind::Poison,
            PowerKind::NoxiousFumes,
            PowerKind::NoBlock,
            PowerKind::Slow,
            PowerKind::Doom,
            // sts2-rs-t88vp: C# PowerType.Debuff variants formerly UNKNOWN
            PowerKind::Constrict,
            PowerKind::Tangled,
            PowerKind::Hex,
            PowerKind::Strangle,
            PowerKind::Debilitate,
            PowerKind::DarkShackles,
            PowerKind::EnfeeblingTouch,
            PowerKind::PiercingWail,
            PowerKind::WraithForm,
            // The variant the Necrobinder SicEm card actually applies
            // (`SicEmPower`, Type=Debuff) — Artifact must negate it.
            PowerKind::SicEmPower,
        ] {
            assert!(classify(k).is_debuff(), "{k:?} should be Debuff");
            assert!(is_affliction(k));
        }
    }

    #[test]
    fn known_buffs_are_buffs() {
        for k in [
            PowerKind::Strength,
            PowerKind::Dexterity,
            PowerKind::Vigor,
            PowerKind::Artifact,
            PowerKind::Intangible,
            PowerKind::Thorns,
            PowerKind::Plating,
            PowerKind::DemonForm,
        ] {
            assert!(classify(k).is_buff(), "{k:?} should be Buff");
            assert!(!is_affliction(k));
        }
    }

    #[test]
    fn type_for_amount_flips_negative_attribute_powers_to_debuff() {
        // C# GetTypeForAmount: Counter+AllowNegative attr (Strength/Dex/Focus)
        // is a Debuff while negative, a Buff at >= 0.
        for k in [PowerKind::Strength, PowerKind::Dexterity, PowerKind::Focus] {
            assert!(is_debuff_for_amount(k, -1), "{k:?}(-1) should read Debuff");
            assert!(is_debuff_for_amount(k, -5), "{k:?}(-5) should read Debuff");
            assert!(!is_debuff_for_amount(k, 3), "{k:?}(3) should read Buff");
            // amount == 0 keeps the static (Buff) type — power would be removed
            // anyway (ShouldRemoveDueToAmount), so it never reaches a counter.
            assert!(!is_debuff_for_amount(k, 0), "{k:?}(0) should read Buff");
        }
        // A static debuff stays a debuff at any positive amount, and a
        // genuinely-debuff Counter (Shriek/Shrink) reads Debuff even negative.
        assert!(is_debuff_for_amount(PowerKind::Vulnerable, 2));
        assert!(is_debuff_for_amount(PowerKind::Shriek, -2));
        assert!(is_debuff_for_amount(PowerKind::Shrink, -1));
        // Plain buffs never flip.
        assert!(!is_debuff_for_amount(PowerKind::Thorns, 5));
    }

    #[test]
    fn coverage_counts_total_matches_enum_size() {
        let c = coverage_counts();
        assert_eq!(c.total, PowerKind::all().len());
        assert_eq!(
            c.buffs + c.debuffs + c.mixed + c.neutrals + c.unknown,
            c.total,
        );
    }
}
