//! Neow event option-roll RNG algorithm. Sub-bead `sts2-rs-qq9og`
//! (R3.1.1 under umbrella `sts2-rs-nr8l4`).
//!
//! Port pin:
//! `../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Events/Neow.cs:213-296`
//! (`Neow.GenerateInitialOptions`, no-modifiers branch).
//!
//! Pure function: [`roll_neow_offers`] takes a per-event RNG (seeded
//! per `EventModel.cs:193` — `RunState.Rng.Seed + NetId +
//! deterministic_hash_code(Id.Entry)`; wiring lives in sub-bead
//! `sts2-rs-14cid`) and returns the 3 relic offers, RNG-faithful to
//! the C# algorithm (same `next_int` / `next_bool` / `shuffle` call
//! sequence).
//!
//! The relic IDs are returned as [`NeowRelic`], a closed enum mirroring
//! the C# pool. Every variant now maps to a first-class [`RelicId`]:
//! `PhialHolster` / `WingedBoots` / `NeowsTalisman` (bead `sts2-rs-4rwv4`,
//! W9a-c), `NeowsBones` (bead `sts2-rs-wjuoi`, W9e), and `HeftyTablet`
//! (bead `sts2-rs-w6hdj`, W9d — 3-Rare card-select + always-add Injury
//! via the run-side pending-card-select surface).

use crate::DotNetRng as Rng;
use crate::RelicId;

/// Closed enum of every relic that can appear in a Neow offer.
/// Mirrors the C# pool at `Neow.cs:67-103`. Every variant resolves to a
/// first-class [`RelicId`] via [`Self::to_relic_id`] (W9a-e).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NeowRelic {
    // ── Positive base pool (v0.107.1 Neow.cs:66-82) ───────────────────
    ArcaneScroll,
    BoomingConch,
    GoldenPearl,
    LeadPaperweight,
    LostCoffer,
    NeowsTorment,
    NewLeaf,
    PreciseScissors,
    PhialHolster,
    WingedBoots,
    MassiveScroll,
    /// v0.107.1 (bead `sts2-rs-v0107-content-port-wb8lk.6`): FishingRod
    /// and Kaleidoscope joined the positive pool, and `ScrollBoxes`
    /// moved here from the v0.103.2 curse-pool add-in slot.
    FishingRod,
    Kaleidoscope,
    // ── Positive 50/50 add-ins (Neow.cs:245-271) ──────────────────────
    LavaRock,
    SmallCapsule,
    NutritiousOyster,
    StoneHumidifier,
    NeowsTalisman,
    Pomander,
    // ── Curse pool (v0.107.1 Neow.cs:96-106) ──────────────────────────
    CursedPearl,
    HeftyTablet,
    LargeCapsule,
    LeafyPoultice,
    PrecariousShears,
    SilverCrucible,
    NeowsBones,
    /// v0.107.1 (bead `sts2-rs-v0107-content-port-wb8lk.6`): new curse
    /// option. On pickup loses all gold; the first card reward after is
    /// Glam-enchanted (one-time).
    SilkenTress,
    /// v0.107.1: `ScrollBoxes` moved out of the curse pool into the
    /// positive pool (it is a static `PositiveOptions` member now). The
    /// variant stays for exhaustive pattern-matching.
    ScrollBoxes,
    // ── v0.111.0 curse-pool additions (`Neow.cs:96-108`, GH #792) ─────
    /// v0.111.0: new curse option — adds the Dowsing quest card.
    DowsingRod,
    /// v0.111.0: new curse option — procures Ambergris + adds Guilty, and
    /// strips PhialHolster / LostCoffer from the positive offers.
    NeowsSacrifice,
}

impl NeowRelic {
    /// Every variant, in declaration order.
    pub const ALL: [Self; 30] = [
        Self::ArcaneScroll,
        Self::BoomingConch,
        Self::GoldenPearl,
        Self::LeadPaperweight,
        Self::LostCoffer,
        Self::NeowsTorment,
        Self::NewLeaf,
        Self::PreciseScissors,
        Self::PhialHolster,
        Self::WingedBoots,
        Self::MassiveScroll,
        Self::FishingRod,
        Self::Kaleidoscope,
        Self::LavaRock,
        Self::SmallCapsule,
        Self::NutritiousOyster,
        Self::StoneHumidifier,
        Self::NeowsTalisman,
        Self::Pomander,
        Self::CursedPearl,
        Self::HeftyTablet,
        Self::LargeCapsule,
        Self::LeafyPoultice,
        Self::PrecariousShears,
        Self::SilverCrucible,
        Self::NeowsBones,
        Self::SilkenTress,
        Self::ScrollBoxes,
        Self::DowsingRod,
        Self::NeowsSacrifice,
    ];

    /// The variant whose [`Self::snake_case_id`] is `id` (the game's relic
    /// `Id.Entry`, e.g. `"MASSIVE_SCROLL"`), or `None`.
    #[must_use]
    pub fn from_snake_case_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|r| r.snake_case_id() == id)
    }

    /// Convert to the matching [`RelicId`]. Every Neow-offer variant now
    /// has a first-class `RelicId` counterpart (W9a-e); the return type
    /// stays `Option` for call-site stability and forward-compat with any
    /// future Neow option that lands ahead of its `RelicId`.
    #[must_use]
    pub fn to_relic_id(self) -> Option<RelicId> {
        Some(match self {
            Self::ArcaneScroll => RelicId::ArcaneScroll,
            Self::BoomingConch => RelicId::BoomingConch,
            Self::GoldenPearl => RelicId::GoldenPearl,
            Self::LeadPaperweight => RelicId::LeadPaperweight,
            Self::LostCoffer => RelicId::LostCoffer,
            Self::NeowsTorment => RelicId::NeowsTorment,
            Self::NewLeaf => RelicId::NewLeaf,
            Self::PreciseScissors => RelicId::PreciseScissors,
            Self::MassiveScroll => RelicId::MassiveScroll,
            Self::LavaRock => RelicId::LavaRock,
            Self::SmallCapsule => RelicId::SmallCapsule,
            Self::NutritiousOyster => RelicId::NutritiousOyster,
            Self::StoneHumidifier => RelicId::StoneHumidifier,
            Self::Pomander => RelicId::Pomander,
            Self::CursedPearl => RelicId::CursedPearl,
            Self::LargeCapsule => RelicId::LargeCapsule,
            Self::LeafyPoultice => RelicId::LeafyPoultice,
            Self::PrecariousShears => RelicId::PrecariousShears,
            Self::SilverCrucible => RelicId::SilverCrucible,
            Self::ScrollBoxes => RelicId::ScrollBoxes,
            Self::PhialHolster => RelicId::PhialHolster,
            Self::WingedBoots => RelicId::WingedBoots,
            Self::NeowsTalisman => RelicId::NeowsTalisman,
            Self::NeowsBones => RelicId::NeowsBones,
            Self::HeftyTablet => RelicId::HeftyTablet,
            Self::FishingRod => RelicId::FishingRod,
            Self::Kaleidoscope => RelicId::Kaleidoscope,
            Self::SilkenTress => RelicId::SilkenTress,
            Self::DowsingRod => RelicId::DowsingRod,
            Self::NeowsSacrifice => RelicId::NeowsSacrifice,
        })
    }

    /// SCREAMING_SNAKE_CASE label matching the scraped-corpus
    /// `relic_id` field. Stable across versions; the 5 missing relics
    /// emit their canonical SCREAMING_SNAKE_CASE name so parity dumps
    /// can compare against the scraped JSON before `RelicId` catches up.
    #[must_use]
    pub fn snake_case_id(self) -> &'static str {
        match self {
            Self::ArcaneScroll => "ARCANE_SCROLL",
            Self::BoomingConch => "BOOMING_CONCH",
            Self::GoldenPearl => "GOLDEN_PEARL",
            Self::LeadPaperweight => "LEAD_PAPERWEIGHT",
            Self::LostCoffer => "LOST_COFFER",
            Self::NeowsTorment => "NEOWS_TORMENT",
            Self::NewLeaf => "NEW_LEAF",
            Self::PreciseScissors => "PRECISE_SCISSORS",
            Self::PhialHolster => "PHIAL_HOLSTER",
            Self::WingedBoots => "WINGED_BOOTS",
            Self::MassiveScroll => "MASSIVE_SCROLL",
            Self::LavaRock => "LAVA_ROCK",
            Self::SmallCapsule => "SMALL_CAPSULE",
            Self::NutritiousOyster => "NUTRITIOUS_OYSTER",
            Self::StoneHumidifier => "STONE_HUMIDIFIER",
            Self::NeowsTalisman => "NEOWS_TALISMAN",
            Self::Pomander => "POMANDER",
            Self::CursedPearl => "CURSED_PEARL",
            Self::HeftyTablet => "HEFTY_TABLET",
            Self::LargeCapsule => "LARGE_CAPSULE",
            Self::LeafyPoultice => "LEAFY_POULTICE",
            Self::PrecariousShears => "PRECARIOUS_SHEARS",
            Self::SilverCrucible => "SILVER_CRUCIBLE",
            Self::NeowsBones => "NEOWS_BONES",
            Self::ScrollBoxes => "SCROLL_BOXES",
            Self::FishingRod => "FISHING_ROD",
            Self::Kaleidoscope => "KALEIDOSCOPE",
            Self::SilkenTress => "SILKEN_TRESS",
            Self::DowsingRod => "DOWSING_ROD",
            Self::NeowsSacrifice => "NEOWS_SACRIFICE",
        }
    }
}

/// The 3 offers Neow presents at run-start: 2 positive relics + 1
/// curse. Mirrors C# `Neow.GenerateInitialOptions` return shape
/// (`[positive_1, positive_2, cursed]` — the curse is always the third
/// option in the UI).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeowOffers {
    pub positive_1: NeowRelic,
    pub positive_2: NeowRelic,
    pub cursed: NeowRelic,
}

/// Base positive pool — Neow.cs:67-80. Listed in the C# array as 11
/// entries including `MassiveScroll`, but `MassiveScroll.IsAllowed`
/// (`MassiveScroll.cs:19`) gates on `runState.Players.Count > 1` so
/// it is filtered out by the `RemoveAll(!IsAllowed)` pass at
/// Neow.cs:225-227 / :272-276 in any single-player run. Since
/// `sts2-rs` only models single-player at this pin (and the entire
/// run-loop assumes `players: [Ironclad]`), we drop `MassiveScroll`
/// from the static pool directly. (Should multi-player land later,
/// gate it on `state.players.len() > 1` here.)
///
/// **v0.107.1** (bead `sts2-rs-v0107-content-port-wb8lk.6`): the C#
/// `PositiveOptions` array grew 11→14 — `FishingRod`, `Kaleidoscope`,
/// and `ScrollBoxes` (moved in from the v0.103.2 curse add-in). Listed
/// here in the exact C# array order (`Neow.cs:66-82`) so the
/// `next_item` / `UnstableShuffle` RNG sequence stays oracle-faithful,
/// minus `MassiveScroll` (single-player `IsAllowedAtNeow == false`).
/// `Kaleidoscope.IsAllowedAtNeow` additionally requires every character
/// card pool unlocked; the run-start sim models a fully-unlocked roster,
/// so it is retained here (matching the v0.107.1 oracle).
const POSITIVE_POOL: &[NeowRelic] = &[
    NeowRelic::ArcaneScroll,
    NeowRelic::BoomingConch,
    NeowRelic::FishingRod,
    NeowRelic::GoldenPearl,
    NeowRelic::Kaleidoscope,
    NeowRelic::LeadPaperweight,
    NeowRelic::LostCoffer,
    NeowRelic::NeowsTorment,
    NeowRelic::NewLeaf,
    NeowRelic::PhialHolster,
    NeowRelic::PreciseScissors,
    NeowRelic::ScrollBoxes,
    NeowRelic::WingedBoots,
];

/// Curse pool — **v0.107.1** `Neow.cs:96-106` `CurseOptions[8]`, in the
/// exact C# array order so the `next_item` curse draw is oracle-faithful
/// (bead `sts2-rs-v0107-content-port-wb8lk.6`). The array grew 7→8 with
/// the new `SilkenTress`; `ScrollBoxes` left the curse pool (it is now a
/// static `PositiveOptions` member — see [`POSITIVE_POOL`]). All eight
/// are unconditional `RelicOption`s here: in v0.103.2 `ScrollBoxes` was
/// the only conditional curse add-in, and it no longer participates.
const CURSE_POOL: &[NeowRelic] = &[
    NeowRelic::CursedPearl,
    NeowRelic::HeftyTablet,
    NeowRelic::LargeCapsule,
    NeowRelic::LeafyPoultice,
    NeowRelic::NeowsBones,
    NeowRelic::PrecariousShears,
    NeowRelic::SilkenTress,
    NeowRelic::SilverCrucible,
];

/// Curse pool — **v0.111.0** `Neow.cs:96-108` `CurseOptions[10]`, in the
/// exact C# array order (the `next_item` curse draw indexes it): the
/// v0.107.1 eight plus `DowsingRod` (after `CursedPearl`) and
/// `NeowsSacrifice` (after `NeowsBones`). Neither new relic overrides
/// `IsAllowedAtNeow`/`IsAllowed` (`DowsingRod.cs`, `NeowsSacrifice.cs`), so
/// both stay in the single-player pool unconditionally.
const CURSE_POOL_V0111: &[NeowRelic] = &[
    NeowRelic::CursedPearl,
    NeowRelic::DowsingRod,
    NeowRelic::HeftyTablet,
    NeowRelic::LargeCapsule,
    NeowRelic::LeafyPoultice,
    NeowRelic::NeowsBones,
    NeowRelic::NeowsSacrifice,
    NeowRelic::PrecariousShears,
    NeowRelic::SilkenTress,
    NeowRelic::SilverCrucible,
];

/// The Neow curse pool for game build `version`.
#[must_use]
pub fn curse_pool_for(version: crate::GameVersion) -> &'static [NeowRelic] {
    match version {
        crate::GameVersion::V0_107_1 => CURSE_POOL,
        // v0.111.0: CurseOptions grew 8 -> 10 (decomp Neow.cs:96).
        crate::GameVersion::V0_111_0 | crate::GameVersion::V0_113_0 => CURSE_POOL_V0111,
    }
}

/// Inline `Rng.NextItem(IList<T>)` — uniform pick over a non-empty
/// slice. C# `Rng.NextItem` calls `NextInt(0, list.Count - 1)`
/// (inclusive both ends); our `next_int(low, high)` matches that
/// inclusive contract (see `dotnet_rng.rs::Rng::next_int`).
fn next_item<T: Copy>(rng: &mut Rng, lst: &[T]) -> T {
    assert!(!lst.is_empty(), "next_item on empty slice");
    let last = i32::try_from(lst.len() - 1).expect("pool size fits in i32");
    let idx = rng.next_int(0, last);
    lst[idx as usize]
}

/// Port of C# `Neow.GenerateInitialOptions` (no-modifiers branch).
///
/// Caller supplies a per-event RNG seeded per
/// `EventModel.cs:193`. The function performs exactly the same RNG
/// operations as the C# code:
///
/// 1. `next_int` to pick one curse from [`CURSE_POOL`].
/// 2. Up to 3 `next_bool` calls to choose between paired add-ins
///    (LavaRock/SmallCapsule — *suppressed* iff `cursed == LargeCapsule*,
///    NutritiousOyster/StoneHumidifier, NeowsTalisman/Pomander).
/// 3. `shuffle` (== C# `UnstableShuffle`) on the positive pool.
///
/// Returns 2 positives + the curse. The function does NOT apply a runtime
/// `IsAllowedAtNeow` filter pass: v0.107.1 renamed the Neow gate
/// `IsAllowed` → `IsAllowedAtNeow` (`Neow.cs:222,272`,
/// `RelicModel.cs:444`), but for the SINGLE-PLAYER, fully-unlocked sim the
/// gate is statically resolved into the pool composition above —
///   * `MassiveScroll.IsAllowed == (Players > 1)` ⇒ false ⇒ pre-dropped from
///     [`POSITIVE_POOL`] (it would be removed by the RemoveAll-before-shuffle
///     pass anyway, so the shuffle RNG sequence is unaffected);
///   * `WingedBoots` / `SilverCrucible` `IsAllowed == (Players == 1)` ⇒ true ⇒
///     retained;
///   * `Kaleidoscope.IsAllowedAtNeow` requires every character card pool
///     unlocked, and `ScrollBoxes.IsAllowedAtNeow` requires
///     `CanGenerateBundles` (≥4 commons + ≥2 uncommons) — both true for the
///     fully-unlocked roster + every shipped character pool, so retained.
/// No remaining pool relic gates. If a future addition introduces a Neow
/// relic that is gated in single-player, add a `.retain(...)` pass between
/// pool construction and `next_item`/`shuffle` (curse pool filtered BEFORE the
/// curse `next_item`, positives AFTER the add-ins and BEFORE the shuffle, per
/// `Neow.cs:218-223` / `:269-275`).
#[must_use]
pub fn roll_neow_offers(rng: &mut Rng) -> NeowOffers {
    roll_neow_offers_for(crate::GameVersion::V0_107_1, rng)
}

/// [`roll_neow_offers`] for game build `version` (GH #792). v0.107.1 is the
/// function above, bit for bit; v0.111.0 draws the curse from the 10-entry
/// pool ([`curse_pool_for`]) and, when it lands on `NeowsSacrifice`, drops
/// `PhialHolster` and `LostCoffer` from the positive pool before the add-ins
/// and the shuffle (`Neow.cs:244-248`):
///
/// [decompiled C# omitted — see the cited path]
#[must_use]
pub fn roll_neow_offers_for(version: crate::GameVersion, rng: &mut Rng) -> NeowOffers {
    // (1) curse pick — Neow.cs:217 + :227.
    let cursed = next_item(rng, curse_pool_for(version));

    // (2) base positive pool, drop curse↔positive conflicts
    //     (Neow.cs:228-244).
    let mut positive_pool: Vec<NeowRelic> = POSITIVE_POOL.to_vec();
    match cursed {
        NeowRelic::CursedPearl => positive_pool.retain(|r| *r != NeowRelic::GoldenPearl),
        NeowRelic::HeftyTablet => positive_pool.retain(|r| *r != NeowRelic::ArcaneScroll),
        NeowRelic::LeafyPoultice => positive_pool.retain(|r| *r != NeowRelic::NewLeaf),
        NeowRelic::PrecariousShears => {
            positive_pool.retain(|r| *r != NeowRelic::PreciseScissors);
        }
        // v0.111.0: NeowsSacrifice strips the PhialHolster / LostCoffer
        // offers (decomp Neow.cs:244). Unreachable in v0.107.1 (not in its
        // curse pool).
        NeowRelic::NeowsSacrifice => {
            positive_pool.retain(|r| *r != NeowRelic::PhialHolster && *r != NeowRelic::LostCoffer);
        }
        _ => {}
    }

    // (3) LavaRock vs SmallCapsule — suppressed iff cursed is
    //     LargeCapsule (Neow.cs:245-255). NB: the C# branch consumes
    //     ONE `NextBool` only when the conditional fires. The Rust
    //     translation must do the same — DO NOT call `next_bool()`
    //     unconditionally.
    if cursed != NeowRelic::LargeCapsule {
        positive_pool.push(if rng.next_bool() {
            NeowRelic::LavaRock
        } else {
            NeowRelic::SmallCapsule
        });
    }

    // (4) NutritiousOyster vs StoneHumidifier — unconditional
    //     (Neow.cs:256-263).
    positive_pool.push(if rng.next_bool() {
        NeowRelic::NutritiousOyster
    } else {
        NeowRelic::StoneHumidifier
    });

    // (5) NeowsTalisman vs Pomander — unconditional (Neow.cs:264-271).
    positive_pool.push(if rng.next_bool() {
        NeowRelic::NeowsTalisman
    } else {
        NeowRelic::Pomander
    });

    // (6) UnstableShuffle (Neow.cs:278). `Rng::shuffle` is exactly
    //     UnstableShuffle (Fisher-Yates with `next_int(0, idx)`); see
    //     `dotnet_rng.rs::Rng::shuffle` doc-comment.
    // v0.113.0: one parent `NextUnsignedLong` seeds the shuffle's child Rng.
    rng.shuffle_for(version, &mut positive_pool);

    NeowOffers {
        positive_1: positive_pool[0],
        positive_2: positive_pool[1],
        cursed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_neow_relic_round_trips_through_its_id() {
        let mut seen = std::collections::HashSet::new();
        for r in NeowRelic::ALL {
            assert!(seen.insert(r), "{r:?} listed twice");
            assert_eq!(NeowRelic::from_snake_case_id(r.snake_case_id()), Some(r));
        }
        assert_eq!(NeowRelic::from_snake_case_id("NOT_A_RELIC"), None);
    }

    #[test]
    fn same_seed_produces_same_offers() {
        let mut a = Rng::with_name(12345, "Neow");
        let mut b = Rng::with_name(12345, "Neow");
        assert_eq!(roll_neow_offers(&mut a), roll_neow_offers(&mut b));
    }

    #[test]
    fn cursed_always_drawn_from_curse_pool() {
        for seed in 0..200 {
            let mut rng = Rng::with_name(seed, "Neow");
            let o = roll_neow_offers(&mut rng);
            assert!(
                CURSE_POOL.contains(&o.cursed),
                "seed {seed}: cursed = {:?}",
                o.cursed
            );
        }
    }

    #[test]
    fn positives_disjoint_from_curse() {
        // Positive and curse pools share no variants; both positives
        // must be from the positive side (base pool or one of the 3
        // 50/50 add-ins).
        let positive_universe: &[NeowRelic] = &[
            NeowRelic::ArcaneScroll,
            NeowRelic::BoomingConch,
            NeowRelic::FishingRod,
            NeowRelic::GoldenPearl,
            NeowRelic::Kaleidoscope,
            NeowRelic::LeadPaperweight,
            NeowRelic::LostCoffer,
            NeowRelic::NeowsTorment,
            NeowRelic::NewLeaf,
            NeowRelic::PreciseScissors,
            NeowRelic::PhialHolster,
            NeowRelic::ScrollBoxes,
            NeowRelic::WingedBoots,
            NeowRelic::MassiveScroll,
            NeowRelic::LavaRock,
            NeowRelic::SmallCapsule,
            NeowRelic::NutritiousOyster,
            NeowRelic::StoneHumidifier,
            NeowRelic::NeowsTalisman,
            NeowRelic::Pomander,
        ];
        for seed in 0..200 {
            let mut rng = Rng::with_name(seed, "Neow");
            let o = roll_neow_offers(&mut rng);
            assert!(positive_universe.contains(&o.positive_1));
            assert!(positive_universe.contains(&o.positive_2));
        }
    }

    #[test]
    fn three_offers_all_distinct() {
        for seed in 0..1000 {
            let mut rng = Rng::with_name(seed, "Neow");
            let o = roll_neow_offers(&mut rng);
            assert_ne!(o.positive_1, o.positive_2, "seed {seed}");
            assert_ne!(o.positive_1, o.cursed, "seed {seed}");
            assert_ne!(o.positive_2, o.cursed, "seed {seed}");
        }
    }

    #[test]
    fn conflict_pairs_never_co_occur() {
        // Neow.cs:228-244 removes the conflicting positive when the
        // curse is rolled. After 5000 seeds we should see at least one
        // hit on each pair (pigeon-hole), and zero violations.
        let mut saw_cursed_pearl = false;
        let mut saw_hefty_tablet = false;
        let mut saw_leafy = false;
        let mut saw_shears = false;
        for seed in 0..5000 {
            let mut rng = Rng::with_name(seed, "Neow");
            let o = roll_neow_offers(&mut rng);
            match o.cursed {
                NeowRelic::CursedPearl => {
                    saw_cursed_pearl = true;
                    assert_ne!(o.positive_1, NeowRelic::GoldenPearl);
                    assert_ne!(o.positive_2, NeowRelic::GoldenPearl);
                }
                NeowRelic::HeftyTablet => {
                    saw_hefty_tablet = true;
                    assert_ne!(o.positive_1, NeowRelic::ArcaneScroll);
                    assert_ne!(o.positive_2, NeowRelic::ArcaneScroll);
                }
                NeowRelic::LeafyPoultice => {
                    saw_leafy = true;
                    assert_ne!(o.positive_1, NeowRelic::NewLeaf);
                    assert_ne!(o.positive_2, NeowRelic::NewLeaf);
                }
                NeowRelic::PrecariousShears => {
                    saw_shears = true;
                    assert_ne!(o.positive_1, NeowRelic::PreciseScissors);
                    assert_ne!(o.positive_2, NeowRelic::PreciseScissors);
                }
                _ => {}
            }
        }
        assert!(saw_cursed_pearl);
        assert!(saw_hefty_tablet);
        assert!(saw_leafy);
        assert!(saw_shears);
    }

    #[test]
    fn large_capsule_suppresses_lava_rock_and_small_capsule() {
        // Neow.cs:245-255: if cursed == LargeCapsule, the LavaRock/
        // SmallCapsule add-in is skipped. Neither should appear among
        // the positives in that case.
        let mut saw = false;
        for seed in 0..5000 {
            let mut rng = Rng::with_name(seed, "Neow");
            let o = roll_neow_offers(&mut rng);
            if o.cursed == NeowRelic::LargeCapsule {
                saw = true;
                assert_ne!(o.positive_1, NeowRelic::LavaRock);
                assert_ne!(o.positive_2, NeowRelic::LavaRock);
                assert_ne!(o.positive_1, NeowRelic::SmallCapsule);
                assert_ne!(o.positive_2, NeowRelic::SmallCapsule);
            }
        }
        assert!(saw, "no seed in 0..5000 rolled LargeCapsule curse");
    }

    #[test]
    fn to_relic_id_some_for_supported_variants() {
        assert_eq!(
            NeowRelic::ArcaneScroll.to_relic_id(),
            Some(RelicId::ArcaneScroll)
        );
        assert_eq!(
            NeowRelic::CursedPearl.to_relic_id(),
            Some(RelicId::CursedPearl)
        );
        assert_eq!(NeowRelic::LavaRock.to_relic_id(), Some(RelicId::LavaRock));
    }

    /// v0.107.1 `Neow.cs:66-82` `PositiveOptions[14]` in C# array order, minus
    /// `MassiveScroll` (single-player `IsAllowed`/`IsAllowedAtNeow == false` —
    /// removed by the RemoveAll pass before the shuffle, so dropping it from
    /// the static pool is RNG-faithful). The 3 v0.107.1 additions
    /// (FishingRod, Kaleidoscope, ScrollBoxes) must sit at their decomp indices.
    #[test]
    fn positive_pool_matches_v0107_decomp_order() {
        assert_eq!(
            POSITIVE_POOL,
            &[
                NeowRelic::ArcaneScroll,
                NeowRelic::BoomingConch,
                NeowRelic::FishingRod,
                NeowRelic::GoldenPearl,
                NeowRelic::Kaleidoscope,
                NeowRelic::LeadPaperweight,
                NeowRelic::LostCoffer,
                NeowRelic::NeowsTorment,
                NeowRelic::NewLeaf,
                NeowRelic::PhialHolster,
                NeowRelic::PreciseScissors,
                NeowRelic::ScrollBoxes,
                NeowRelic::WingedBoots,
            ],
        );
        // ScrollBoxes is a POSITIVE member in v0.107.1 (was a v0.103.2 curse
        // add-in); it must NOT appear in the curse pool.
        assert!(!CURSE_POOL.contains(&NeowRelic::ScrollBoxes));
        assert!(POSITIVE_POOL.contains(&NeowRelic::ScrollBoxes));
        // MassiveScroll is single-player-gated → never in the rolled pool.
        assert!(!POSITIVE_POOL.contains(&NeowRelic::MassiveScroll));
    }

    /// v0.107.1 `Neow.cs:96-106` `CurseOptions[8]` in C# array order. The 7→8
    /// growth adds `SilkenTress`; `NeowsBones` sorts up into alphabetical slot.
    #[test]
    fn curse_pool_matches_v0107_decomp_order() {
        assert_eq!(
            CURSE_POOL,
            &[
                NeowRelic::CursedPearl,
                NeowRelic::HeftyTablet,
                NeowRelic::LargeCapsule,
                NeowRelic::LeafyPoultice,
                NeowRelic::NeowsBones,
                NeowRelic::PrecariousShears,
                NeowRelic::SilkenTress,
                NeowRelic::SilverCrucible,
            ],
        );
    }

    /// The three relics new to the v0.107.1 Neow pools are present + map to
    /// their RelicIds (guards the `to_relic_id` wiring for the additions).
    #[test]
    fn v0107_new_neow_relics_present_and_mapped() {
        assert!(POSITIVE_POOL.contains(&NeowRelic::FishingRod));
        assert!(POSITIVE_POOL.contains(&NeowRelic::Kaleidoscope));
        assert!(CURSE_POOL.contains(&NeowRelic::SilkenTress));
        assert_eq!(
            NeowRelic::FishingRod.to_relic_id(),
            Some(RelicId::FishingRod)
        );
        assert_eq!(
            NeowRelic::Kaleidoscope.to_relic_id(),
            Some(RelicId::Kaleidoscope)
        );
        assert_eq!(
            NeowRelic::SilkenTress.to_relic_id(),
            Some(RelicId::SilkenTress)
        );
    }

    // v0.107.1 RNG rework: anchored to a SCRAPED v0.103.2 human run
    // (System.Random). Under v0.107.1's MegaRandom the same seed yields different
    // Neow offers, so this v0.103.2 corpus anchor cannot hold on this branch (the
    // run corpus is build_id==v0.103.2; no v0.107.1 corpus exists yet).
    #[ignore = "v0.103.2 scraped-corpus anchor; invalid under v0.107.1 MegaRandom RNG (no v0.107.1 corpus yet)"]
    #[test]
    fn corpus_seed_2j2dekxjp9_matches_scraped_offers() {
        // Bead `sts2-rs-k8z6m` corpus-parity anchor. Scraped real-game
        // run at `sts2_run_scraper/data/files/the-ironclad/2026-05/
        // 05d33913-6a09-4a79-aacd-9d1409af18eb.run`:
        //   seed = "2J2DEKXJP9", ancient_choice =
        //   [PHIAL_HOLSTER, LOST_COFFER, LEAFY_POULTICE].
        //
        // The full RunState wiring (NetId offset, RNG salt "NEOW") is
        // exercised via the `RunState::roll_neow_offers_if_needed`
        // path, not just `roll_neow_offers` here, so this test imports
        // the helper from runstate's seed convention. We mirror the
        // canonical-seed → hash pipeline inline to keep this test in
        // sts2-core (no sts2-content dep).
        let canonical = crate::canonicalize_seed("2J2DEKXJP9");
        let master = crate::deterministic_hash_code(&canonical);
        // Reconstruct the same Rng the RunState path uses: master+NetId,
        // then mix the "NEOW" salt via with_name. Pins:
        // `EventModel.cs:193`, `NGame.cs:884` (NetId=1uL in SP).
        const NET_ID: i32 = 1;
        let mut rng = Rng::with_name(master.wrapping_add(NET_ID), "NEOW");
        let offers = roll_neow_offers(&mut rng);
        assert_eq!(offers.positive_1, NeowRelic::PhialHolster);
        assert_eq!(offers.positive_2, NeowRelic::LostCoffer);
        assert_eq!(offers.cursed, NeowRelic::LeafyPoultice);
    }

    #[test]
    fn to_relic_id_maps_w9a_e_variants() {
        // Bead `sts2-rs-4rwv4` (W9a-c) + `sts2-rs-wjuoi` (W9e) +
        // `sts2-rs-w6hdj` (W9d): the last five C#-only Neow relics.
        assert_eq!(
            NeowRelic::PhialHolster.to_relic_id(),
            Some(RelicId::PhialHolster)
        );
        assert_eq!(
            NeowRelic::WingedBoots.to_relic_id(),
            Some(RelicId::WingedBoots)
        );
        assert_eq!(
            NeowRelic::NeowsTalisman.to_relic_id(),
            Some(RelicId::NeowsTalisman)
        );
        assert_eq!(
            NeowRelic::NeowsBones.to_relic_id(),
            Some(RelicId::NeowsBones)
        );
        assert_eq!(
            NeowRelic::HeftyTablet.to_relic_id(),
            Some(RelicId::HeftyTablet)
        );
    }

    #[test]
    fn snake_case_ids_match_relic_id_for_overlapping_variants() {
        // Any NeowRelic with a RelicId counterpart must emit the same
        // SCREAMING_SNAKE_CASE string as the RelicId.
        for nr in [
            NeowRelic::ArcaneScroll,
            NeowRelic::BoomingConch,
            NeowRelic::GoldenPearl,
            NeowRelic::LeadPaperweight,
            NeowRelic::LostCoffer,
            NeowRelic::NeowsTorment,
            NeowRelic::NewLeaf,
            NeowRelic::PreciseScissors,
            NeowRelic::MassiveScroll,
            NeowRelic::LavaRock,
            NeowRelic::SmallCapsule,
            NeowRelic::NutritiousOyster,
            NeowRelic::StoneHumidifier,
            NeowRelic::Pomander,
            NeowRelic::CursedPearl,
            NeowRelic::LargeCapsule,
            NeowRelic::LeafyPoultice,
            NeowRelic::PrecariousShears,
            NeowRelic::SilverCrucible,
            NeowRelic::ScrollBoxes,
            NeowRelic::FishingRod,
            NeowRelic::Kaleidoscope,
            NeowRelic::SilkenTress,
        ] {
            let rid = nr.to_relic_id().expect("variant has RelicId");
            assert_eq!(
                nr.snake_case_id(),
                rid.to_python_name(),
                "mismatch for {nr:?}"
            );
        }
    }

    // ── v0.111.0 (GH #792) ───────────────────────────────────────────

    use crate::GameVersion;

    /// `Neow.cs:96-108` CurseOptions[10], in C# array order.
    #[test]
    fn curse_pool_matches_decomp_order_v0111() {
        assert_eq!(
            curse_pool_for(GameVersion::V0_111_0),
            &[
                NeowRelic::CursedPearl,
                NeowRelic::DowsingRod,
                NeowRelic::HeftyTablet,
                NeowRelic::LargeCapsule,
                NeowRelic::LeafyPoultice,
                NeowRelic::NeowsBones,
                NeowRelic::NeowsSacrifice,
                NeowRelic::PrecariousShears,
                NeowRelic::SilkenTress,
                NeowRelic::SilverCrucible,
            ]
        );
        assert_eq!(curse_pool_for(GameVersion::V0_107_1), CURSE_POOL);
    }

    /// The v0.107.1 entry point is the versioned roll at v0.107.1, and the
    /// v0.111.0-only relics never appear in a v0.107.1 offer.
    #[test]
    fn v0107_offers_are_unchanged_and_never_offer_v0111_relics() {
        for seed in 0..500 {
            let mut a = Rng::with_name(seed, "NEOW");
            let mut b = Rng::with_name(seed, "NEOW");
            let o = roll_neow_offers(&mut a);
            assert_eq!(o, roll_neow_offers_for(GameVersion::V0_107_1, &mut b));
            assert_eq!(a, b, "same RNG consumption, seed {seed}");
            for r in [o.positive_1, o.positive_2, o.cursed] {
                assert!(
                    !matches!(r, NeowRelic::DowsingRod | NeowRelic::NeowsSacrifice),
                    "seed {seed}: {r:?}"
                );
            }
        }
    }

    /// `Neow.cs:244-248`: a NeowsSacrifice curse strips PhialHolster and
    /// LostCoffer from the positives; both new curses are reachable.
    #[test]
    fn neows_sacrifice_strips_phial_holster_and_lost_coffer_v0111() {
        let (mut saw_sacrifice, mut saw_dowsing) = (false, false);
        for seed in 0..2000 {
            let mut rng = Rng::with_name(seed, "NEOW");
            let o = roll_neow_offers_for(GameVersion::V0_111_0, &mut rng);
            assert!(curse_pool_for(GameVersion::V0_111_0).contains(&o.cursed));
            saw_dowsing |= o.cursed == NeowRelic::DowsingRod;
            if o.cursed == NeowRelic::NeowsSacrifice {
                saw_sacrifice = true;
                for p in [o.positive_1, o.positive_2] {
                    assert!(
                        !matches!(p, NeowRelic::PhialHolster | NeowRelic::LostCoffer),
                        "seed {seed}: {p:?} offered beside NeowsSacrifice"
                    );
                }
            }
        }
        assert!(saw_sacrifice && saw_dowsing);
    }

    #[test]
    fn v0111_neow_relics_map_to_relic_ids_v0111() {
        assert_eq!(
            NeowRelic::DowsingRod.to_relic_id(),
            Some(RelicId::DowsingRod)
        );
        assert_eq!(
            NeowRelic::NeowsSacrifice.to_relic_id(),
            Some(RelicId::NeowsSacrifice)
        );
        assert_eq!(NeowRelic::DowsingRod.snake_case_id(), "DOWSING_ROD");
        assert_eq!(NeowRelic::NeowsSacrifice.snake_case_id(), "NEOWS_SACRIFICE");
        assert_eq!(RelicId::DowsingRod.introduced_in(), GameVersion::V0_111_0);
        assert!(!RelicId::all().contains(&RelicId::DowsingRod));
        assert!(RelicId::all_for(GameVersion::V0_111_0).contains(&RelicId::NeowsSacrifice));
        assert_eq!(RelicId::all_for(GameVersion::V0_107_1), RelicId::all());
    }
}
