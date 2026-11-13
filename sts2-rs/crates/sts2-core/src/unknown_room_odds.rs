//! Unknown (`?`) map-point room-type resolver — bead `sts2-rs-khzyk`
//! (P9.2.f bounded slice).
//!
//! Ports `sts2-rl-agent@01c094d:sts2_env/run/odds.py::UnknownMapPointOdds`
//! (the rolling-odds pity-counter for `?` rooms) plus the resolver
//! that drives it (`Player.resolve_unknown_room` —
//! `sts2_env/run/run_state.py`, called by
//! `run_manager._enter_room`).
//!
//! ## What this module lands
//!
//! - [`UnknownMapPointOdds`] — the rolling per-room-type odds with
//!   reset-on-roll / grow-on-skip semantics. Boost path (Elite via the
//!   `modify_odds_increase_for_unrolled_room_type` modifier chain) is
//!   deferred — the modifier list isn't ported yet, and Elite starts
//!   at the Python `BASE_ELITE = -1.00` (impossible until boosted) so
//!   the base resolver path never returns Elite anyway.
//! - [`RunState::resolve_unknown_room`] — runs the modify-room-types
//!   channel (bead `sts2-rs-vuxmm`, JuzuBracelet's downstream consumer)
//!   then rolls the odds to produce a concrete [`RoomType`].
//!
//! ## What this module deliberately skips (parity-grade-but-bounded)
//!
//! - **First-run special** (Python `unlock_state.get(NUMBER_OF_RUNS)`
//!   path forcing `EVENT` for the first 2 unknowns + `MONSTER` for the
//!   3rd) — `UnlockState` isn't ported. Will rejoin once
//!   `Player.unlock_state` lands.
//! - **Per-card `modify_unknown_map_point_room_types`** — Python walks
//!   `player.deck` cards before relics. No card today has that hook in
//!   the Rust port; relic-side is the only listener that matters at
//!   the pin.
//! - **`modify_odds_increase_for_unrolled_room_type` modifier chain**
//!   — Python's `RunState.modifiers: list[Modifier]` isn't ported.
//!   This affects only the per-roll growth rate (a 1-2 relic interaction
//!   today); the unmodified growth rate is the parity-default.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::dotnet_rng::Rng;
use crate::runstate::{RoomType, RunState};

/// Python `BASE_MONSTER = 0.10`. Probability `?` rolls a monster fight
/// on first visit; grows by this much per skipped roll.
pub const BASE_MONSTER: f32 = 0.10;
/// Python `BASE_ELITE = -1.00`. Sentinel meaning "impossible until a
/// modifier boosts the odds positive". Today no modifier path lands an
/// Elite via `?` — the field is reserved for parity.
pub const BASE_ELITE: f32 = -1.00;
/// Python `BASE_TREASURE = 0.02`. Probability `?` rolls a treasure
/// chest on first visit.
pub const BASE_TREASURE: f32 = 0.02;
/// Python `BASE_SHOP = 0.03`. Probability `?` rolls a merchant on
/// first visit.
pub const BASE_SHOP: f32 = 0.03;

/// C# room-type ordering used by the fallback selector when
/// `RoomType::Event` is excluded from the candidate set. Lower number
/// = preferred fallback. Mirrors
/// `C_SHARP_ROOM_TYPE_ORDER` in `sts2_env/run/odds.py`.
#[must_use]
pub fn fallback_order(rt: RoomType) -> u8 {
    match rt {
        RoomType::Monster => 1,
        RoomType::Elite => 2,
        RoomType::Boss => 3,
        RoomType::Treasure => 4,
        RoomType::Shop => 5,
        RoomType::Event => 6,
        RoomType::RestSite => 7,
        RoomType::Unknown => u8::MAX,
    }
}

/// Rolling pity-odds container for `?` rooms. One per
/// [`crate::RunState`]. Mirrors Python
/// `sts2_env/run/odds.py::UnknownMapPointOdds`.
///
/// `current` holds the live per-type odds; rolled type → reset to
/// `base`; un-rolled types in the candidate set → grow by `base`.
#[derive(Clone, Debug)]
// Custom `PartialEq` / `Eq` on bit patterns: odds values are built
// from named constants only (no NaN reachable) and mutated by adding
// those same constants, so bitwise comparison is total and lets the
// containing `RunState` keep its derived `Eq`.
pub struct UnknownMapPointOdds {
    /// Live odds per room type — the value compared against the roll.
    pub current: BTreeMap<RoomType, f32>,
    /// Reset / growth base — what each `current[type]` snaps back to
    /// on roll and grows by on skip.
    base: BTreeMap<RoomType, f32>,
}

impl Default for UnknownMapPointOdds {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for UnknownMapPointOdds {
    fn eq(&self, other: &Self) -> bool {
        fn bits_match(a: &BTreeMap<RoomType, f32>, b: &BTreeMap<RoomType, f32>) -> bool {
            if a.len() != b.len() {
                return false;
            }
            a.iter()
                .zip(b.iter())
                .all(|((ka, va), (kb, vb))| ka == kb && va.to_bits() == vb.to_bits())
        }
        bits_match(&self.current, &other.current) && bits_match(&self.base, &other.base)
    }
}

impl Eq for UnknownMapPointOdds {}

impl UnknownMapPointOdds {
    /// Construct a fresh roller with Python's base odds.
    #[must_use]
    pub fn new() -> Self {
        let mut current = BTreeMap::new();
        let mut base = BTreeMap::new();
        for (rt, odds) in [
            (RoomType::Monster, BASE_MONSTER),
            (RoomType::Elite, BASE_ELITE),
            (RoomType::Treasure, BASE_TREASURE),
            (RoomType::Shop, BASE_SHOP),
        ] {
            current.insert(rt, odds);
            base.insert(rt, odds);
        }
        Self { current, base }
    }

    /// Reset all live odds to their base values. Mirrors Python
    /// `UnknownMapPointOdds.reset_to_base()`. Used by tests + the
    /// boss-transition restart path.
    pub fn reset_to_base(&mut self) {
        for (rt, odds) in &self.base {
            self.current.insert(*rt, *odds);
        }
    }

    /// Roll a concrete [`RoomType`] for an `?` room.
    ///
    /// `room_types` is the candidate set after channel-modifications
    /// (see [`RunState::resolve_unknown_room`]). It MUST include
    /// `RoomType::Event` as the default fallback per Python — if no
    /// roll lands, Event is returned.
    ///
    /// Mirrors Python `UnknownMapPointOdds.roll` minus the first-run
    /// `unlock_state` early-out and minus the per-modifier
    /// `modify_odds_increase_for_unrolled_room_type` chain.
    pub fn roll(&mut self, rng: &mut Rng, room_types: &BTreeSet<RoomType>) -> RoomType {
        // Default fallback. Python: if EVENT in room_types → result =
        // EVENT, else result = min(room_types, key=C# order).
        let mut result = if room_types.contains(&RoomType::Event) {
            RoomType::Event
        } else {
            *room_types
                .iter()
                .min_by_key(|rt| fallback_order(**rt))
                .expect("room_types must be non-empty")
        };

        // Cumulative roll over odds[type] for type in room_types where
        // odds >= 0. Python iterates `self._current.items()` (insertion
        // order on a dict). BTreeMap is `RoomType`-ordered (engine
        // enum order: Unknown < Monster < Elite < Boss < Shop <
        // RestSite < Treasure < Event), which differs from Python's
        // insertion order (Monster, Elite, Treasure, Shop). Match
        // Python exactly by iterating in insertion order.
        // C# sums and compares in `float` (UnknownMapPointOdds.cs:140-157).
        let roll_val = rng.next_float_f32();
        let mut cumulative = 0.0_f32;
        for rt in [
            RoomType::Monster,
            RoomType::Elite,
            RoomType::Treasure,
            RoomType::Shop,
        ] {
            if !room_types.contains(&rt) {
                continue;
            }
            let odds = *self.current.get(&rt).unwrap_or(&0.0);
            if odds < 0.0 {
                continue;
            }
            cumulative += odds;
            if roll_val <= cumulative {
                result = rt;
                break;
            }
        }

        // Update odds: rolled type → reset to base; un-rolled types
        // in the candidate set → grow by base. Python applies the
        // modifier chain to the growth term; we use base_odds
        // unmodified (see module docs).
        for (rt, base_odds) in self.base.clone() {
            if rt == result {
                self.current.insert(rt, base_odds);
            } else if room_types.contains(&rt) {
                let cur = *self.current.get(&rt).unwrap_or(&0.0);
                self.current.insert(rt, cur + base_odds);
            }
        }

        result
    }
}

impl RunState {
    /// Whether the deck's LanternKey quest card is active: it is in the deck
    /// and the run is in act 3 (C# `CurrentActIndex == 2`). While it is, the
    /// card turns every `?` into an Event (`ModifyUnknownMapPointRoomTypes`)
    /// and replaces the next event with `WarHistorianRepy`
    /// (`ModifyNextEvent`). Decomp `Models.Cards/LanternKey.cs:21-37`, the
    /// same in v0.107.1 and v0.111.0.
    #[must_use]
    pub fn lantern_key_redirects_to_war_historian(&self) -> bool {
        self.act_index() == 3
            && self
                .master_deck
                .iter()
                .any(|c| c.card_id == crate::cards::CardId::LanternKey)
    }

    /// Resolve an `?` (unknown) map-point room type to a concrete
    /// [`RoomType`]. Mirrors the resolver in Python
    /// `run_manager._enter_room` (called when the player lands on an
    /// `?`).
    ///
    /// Steps:
    /// 1. Build the candidate set
    ///    `{Monster, Treasure, Shop, Event} − blacklist`. Elite is
    ///    excluded by base-odds = -1.00 (no modifier-boost path lands
    ///    today; see module docs).
    /// 2. Walk
    ///    [`crate::run_effects::dispatch_modify_unknown_map_point_room_types`]
    ///    so JuzuBracelet (and any future listener) can drop
    ///    candidates. Channel landed in bead `sts2-rs-vuxmm`.
    /// 3. Roll via [`Self::unknown_map_point_odds`] over the live
    ///    `unknown_map_point` named RNG stream
    ///    (`crate::run_rng_set::RunRngSet::unknown_map_point`).
    ///
    /// Bead `sts2-rs-khzyk` (P9.2.f bounded slice).
    pub fn resolve_unknown_room(&mut self, blacklist: BTreeSet<RoomType>) -> RoomType {
        // Default candidate set: monster + treasure + shop + event,
        // minus blacklist. Elite is intentionally absent — its base
        // odds are -1 (never positive without a modifier-boost) and
        // the modifier chain isn't ported.
        let mut room_types: BTreeSet<RoomType> = [
            RoomType::Monster,
            RoomType::Treasure,
            RoomType::Shop,
            RoomType::Event,
        ]
        .into_iter()
        .filter(|rt| !blacklist.contains(rt))
        .collect();

        // Walk the relic-side channel (JuzuBracelet drops `Monster`
        // pre-act-3-treasure, etc.).
        room_types = self.modify_unknown_map_point_room_types(room_types);
        // The LanternKey quest card's listener on the same hook: in act 3
        // every `?` is an Event (decomp `Models.Cards/LanternKey.cs:21-28`).
        // The other listeners only narrow the set (JuzuBracelet) or also
        // return `{Event}` (GoldenCompass), so their order does not matter.
        if self.lantern_key_redirects_to_war_historian() {
            room_types = std::iter::once(RoomType::Event).collect();
        }
        // v0.113.0 Dowsing quest card: strips Monster/Elite after 3 rooms.
        if self.dowsing_strips_fights() {
            room_types.remove(&RoomType::Monster);
            room_types.remove(&RoomType::Elite);
        }

        // Roll.
        self.unknown_map_point_odds
            .roll(&mut self.rng.unknown_map_point, &room_types)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_rng_set::RunRngSet;

    fn rng(seed: i32) -> Rng {
        let mut set = RunRngSet::new(seed);
        std::mem::replace(&mut set.unknown_map_point, Rng::with_name(0, "scratch"))
    }

    #[test]
    fn new_starts_at_python_base_odds() {
        let odds = UnknownMapPointOdds::new();
        assert!((odds.current[&RoomType::Monster] - BASE_MONSTER).abs() < 1e-12);
        assert!((odds.current[&RoomType::Elite] - BASE_ELITE).abs() < 1e-12);
        assert!((odds.current[&RoomType::Treasure] - BASE_TREASURE).abs() < 1e-12);
        assert!((odds.current[&RoomType::Shop] - BASE_SHOP).abs() < 1e-12);
    }

    #[test]
    fn elite_negative_base_never_rolled_without_boost() {
        let mut odds = UnknownMapPointOdds::new();
        let candidates: BTreeSet<RoomType> = [
            RoomType::Monster,
            RoomType::Elite,
            RoomType::Treasure,
            RoomType::Shop,
            RoomType::Event,
        ]
        .into_iter()
        .collect();
        let mut r = rng(7);
        for _ in 0..200 {
            let got = odds.roll(&mut r, &candidates);
            assert_ne!(got, RoomType::Elite, "Elite must not roll at base odds");
        }
    }

    #[test]
    fn rolled_type_resets_to_base_skipped_types_grow() {
        let mut odds = UnknownMapPointOdds::new();
        // Force a monster roll: crank monster odds so any roll < 1
        // hits monster.
        odds.current.insert(RoomType::Monster, 1.0);
        let candidates: BTreeSet<RoomType> = [
            RoomType::Monster,
            RoomType::Treasure,
            RoomType::Shop,
            RoomType::Event,
        ]
        .into_iter()
        .collect();
        let mut r = rng(11);
        let got = odds.roll(&mut r, &candidates);
        assert_eq!(got, RoomType::Monster);
        // Monster reset to base.
        assert!((odds.current[&RoomType::Monster] - BASE_MONSTER).abs() < 1e-12);
        // Treasure / Shop grew by their base.
        assert!(odds.current[&RoomType::Treasure] > BASE_TREASURE);
        assert!(odds.current[&RoomType::Shop] > BASE_SHOP);
    }

    #[test]
    fn event_is_fallback_when_no_odds_land() {
        let mut odds = UnknownMapPointOdds::new();
        // Wipe positive odds so cumulative never exceeds the roll.
        odds.current.insert(RoomType::Monster, 0.0);
        odds.current.insert(RoomType::Treasure, 0.0);
        odds.current.insert(RoomType::Shop, 0.0);
        let candidates: BTreeSet<RoomType> = [
            RoomType::Monster,
            RoomType::Treasure,
            RoomType::Shop,
            RoomType::Event,
        ]
        .into_iter()
        .collect();
        let mut r = rng(13);
        let got = odds.roll(&mut r, &candidates);
        assert_eq!(got, RoomType::Event);
    }

    #[test]
    fn event_excluded_falls_back_to_c_sharp_min_order() {
        let mut odds = UnknownMapPointOdds::new();
        odds.current.insert(RoomType::Monster, 0.0);
        odds.current.insert(RoomType::Treasure, 0.0);
        odds.current.insert(RoomType::Shop, 0.0);
        // Drop Event; the C# order (Monster=1 < Treasure=4 < Shop=5)
        // makes Monster the fallback.
        let candidates: BTreeSet<RoomType> =
            [RoomType::Monster, RoomType::Treasure, RoomType::Shop]
                .into_iter()
                .collect();
        let mut r = rng(17);
        let got = odds.roll(&mut r, &candidates);
        assert_eq!(got, RoomType::Monster);
    }

    #[test]
    fn reset_to_base_restores_every_room_type() {
        let mut odds = UnknownMapPointOdds::new();
        odds.current.insert(RoomType::Monster, 0.99);
        odds.current.insert(RoomType::Treasure, 0.99);
        odds.reset_to_base();
        assert!((odds.current[&RoomType::Monster] - BASE_MONSTER).abs() < 1e-12);
        assert!((odds.current[&RoomType::Treasure] - BASE_TREASURE).abs() < 1e-12);
        assert!((odds.current[&RoomType::Shop] - BASE_SHOP).abs() < 1e-12);
    }

    #[test]
    fn resolve_unknown_room_honors_blacklist() {
        let mut rs = RunState::default();
        // Push monster odds high so non-blacklisted monster would
        // otherwise win.
        rs.unknown_map_point_odds
            .current
            .insert(RoomType::Monster, 1.0);
        let blacklist: BTreeSet<RoomType> = [RoomType::Monster].into_iter().collect();
        let got = rs.resolve_unknown_room(blacklist);
        assert_ne!(got, RoomType::Monster);
    }

    fn lantern_key() -> crate::cards::Card {
        crate::cards::Card {
            card_id: crate::cards::CardId::LanternKey,
            ..crate::cards::Card::pod_defaults()
        }
    }

    /// LanternKey's `ModifyUnknownMapPointRoomTypes` (LanternKey.cs:21-28):
    /// in act 3 a `?` is always an Event, and the roll still draws its float
    /// without moving the odds (nothing in `{Event}` has odds to reset).
    #[test]
    fn lantern_key_makes_every_act3_unknown_an_event() {
        let mut rs = RunState::new();
        rs.act_index = 3;
        rs.master_deck.push(lantern_key());
        rs.unknown_map_point_odds
            .current
            .insert(RoomType::Monster, 1.0);
        let odds_before = rs.unknown_map_point_odds.clone();
        let counter_before = rs.rng.unknown_map_point.counter();
        assert_eq!(rs.resolve_unknown_room(BTreeSet::new()), RoomType::Event);
        assert_eq!(rs.rng.unknown_map_point.counter(), counter_before + 1);
        assert_eq!(rs.unknown_map_point_odds, odds_before);
    }

    /// Outside act 3 the card does nothing (`2 != CurrentActIndex`).
    #[test]
    fn lantern_key_leaves_other_acts_alone() {
        let mut rs = RunState::new();
        rs.act_index = 2;
        rs.master_deck.push(lantern_key());
        rs.unknown_map_point_odds
            .current
            .insert(RoomType::Monster, 1.0);
        assert_eq!(rs.resolve_unknown_room(BTreeSet::new()), RoomType::Monster);
    }

    /// `UnknownMapPointOdds.Roll` (`UnknownMapPointOdds.cs:140-175`) sums the
    /// odds in `float` and compares a `float` roll with `<=`. Each seed's
    /// first draw sits just above a cumulative boundary (0.1, 0.12, 0.15) in
    /// double but rounds onto it in float, so C# picks the earlier room.
    #[test]
    fn roll_compares_float_cumulative_odds() {
        let candidates: BTreeSet<RoomType> = [
            RoomType::Monster,
            RoomType::Elite,
            RoomType::Treasure,
            RoomType::Shop,
            RoomType::Event,
        ]
        .into_iter()
        .collect();
        for (seed, want) in [
            (59_306_271_u64, RoomType::Monster),
            (16_718_349, RoomType::Treasure),
            (25_847_251, RoomType::Shop),
        ] {
            let mut odds = UnknownMapPointOdds::new();
            let mut r = Rng::from_seed(seed);
            assert_eq!(odds.roll(&mut r, &candidates), want, "seed {seed}");
        }
    }
}
