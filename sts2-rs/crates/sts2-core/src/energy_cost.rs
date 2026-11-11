//! A card's energy cost: C# `CardEnergyCost` (`CardEnergyCost.cs`) with its
//! ordered `LocalCostModifier` list (`LocalCostModifier.cs`). Both files are
//! identical in v0.107.1 and v0.111.0.
//!
//! C# keeps a base cost (`_base`) plus an ORDERED list of local modifiers.
//! Each modifier has an amount, a type (Absolute: "the cost is N"; Relative:
//! "add N"), a reduce-only flag (apply only if it lowers the running cost)
//! and expiry bits (end of turn, when played; neither = end of combat).
//! `GetWithModifiers(Local)` folds them over the base in insertion order and
//! clamps only at the very end, after the global hooks. `EndOfTurnCleanup`
//! and `AfterCardPlayedCleanup` drop the modifiers carrying that bit, and
//! `UpgradeBy` moves the base and clamps every Absolute amount above a
//! lowered base.
//!
//! [`CardEnergyCost`] is that model in a fixed-capacity, `Copy` form (cards
//! live in a memcpy-cloned arena). Modifiers are only ever appended through
//! [`CardEnergyCost::push`], which keeps the list short with rewrites that
//! cannot change any value the list will ever produce (see there).
//!
//! X-cost cards and unplayable cards carry a negative base. C#
//! `GetWithModifiers` returns `_base` for both before reading a modifier
//! (`if (_base < 0) return num; if (CostsX) return num;`), so a modifier on
//! them is never observable and [`CardEnergyCost::push`] drops it.

use core::fmt;
use core::hash::{Hash, Hasher};

/// The most local modifiers a card can hold at once. See
/// [`CardEnergyCost::push`] for why the compacted list stays well below it
/// and what happens if it would not.
pub const MAX_COST_MODS: usize = 8;

/// C# `LocalCostModifierExpiration`, in the four combinations the setters
/// use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CostExpiry {
    /// `EndOfCombat`: the `…ThisCombat` setters.
    Combat,
    /// `EndOfTurn`: `SetThisTurn` / `AddThisTurn`.
    Turn,
    /// `WhenPlayed`: `SetUntilPlayed` / `AddUntilPlayed`.
    Played,
    /// `EndOfTurn | WhenPlayed`: the `…ThisTurnOrUntilPlayed` setters
    /// (`SetToFreeThisTurn`, Snecko Oil, Enlightenment).
    TurnOrPlayed,
}

impl CostExpiry {
    const fn bits(self) -> u8 {
        match self {
            Self::Combat => 0,
            Self::Turn => LocalCostMod::END_OF_TURN,
            Self::Played => LocalCostMod::WHEN_PLAYED,
            Self::TurnOrPlayed => LocalCostMod::END_OF_TURN | LocalCostMod::WHEN_PLAYED,
        }
    }
}

/// One C# `LocalCostModifier`: an amount and a flag byte. `Ord` is a plain
/// total order (amount, then flags) for sorting identity keys.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalCostMod {
    amount: i8,
    flags: u8,
}

impl LocalCostMod {
    /// `LocalCostType.Absolute` (clear: `Relative`).
    pub const ABSOLUTE: u8 = 1;
    /// `IsReduceOnly`.
    pub const REDUCE_ONLY: u8 = 1 << 1;
    /// `LocalCostModifierExpiration.EndOfTurn`.
    pub const END_OF_TURN: u8 = 1 << 2;
    /// `LocalCostModifierExpiration.WhenPlayed`.
    pub const WHEN_PLAYED: u8 = 1 << 3;
    const EXPIRY: u8 = Self::END_OF_TURN | Self::WHEN_PLAYED;
    const ALL: u8 = Self::ABSOLUTE | Self::REDUCE_ONLY | Self::EXPIRY;

    /// What `SetToFreeThisTurn` pushes: `SetThisTurnOrUntilPlayed(0)`.
    pub const FREE_THIS_TURN: Self = Self::absolute(0, CostExpiry::TurnOrPlayed, false);
    /// What `SetToFreeThisCombat` pushes: `SetThisCombat(0)`.
    pub const FREE_THIS_COMBAT: Self = Self::absolute(0, CostExpiry::Combat, false);

    /// The modifier a `Set…` setter pushes ("the cost is `amount`").
    #[must_use]
    pub const fn absolute(amount: i8, expiry: CostExpiry, reduce_only: bool) -> Self {
        let rof = if reduce_only { Self::REDUCE_ONLY } else { 0 };
        Self {
            amount,
            flags: Self::ABSOLUTE | rof | expiry.bits(),
        }
    }

    /// The modifier an `Add…` setter pushes ("add `amount`").
    #[must_use]
    pub const fn relative(amount: i8, expiry: CostExpiry) -> Self {
        Self {
            amount,
            flags: expiry.bits(),
        }
    }

    /// Rebuild a modifier from its [`Self::amount`] and [`Self::flags`] (a
    /// serialized state). `None` for an unknown flag bit.
    #[must_use]
    pub const fn from_parts(amount: i8, flags: u8) -> Option<Self> {
        if flags & !Self::ALL != 0 {
            return None;
        }
        Some(Self { amount, flags })
    }

    #[must_use]
    pub const fn amount(self) -> i8 {
        self.amount
    }

    /// The flag byte: [`Self::ABSOLUTE`], [`Self::REDUCE_ONLY`],
    /// [`Self::END_OF_TURN`], [`Self::WHEN_PLAYED`].
    #[must_use]
    pub const fn flags(self) -> u8 {
        self.flags
    }

    #[must_use]
    pub const fn is_absolute(self) -> bool {
        self.flags & Self::ABSOLUTE != 0
    }

    #[must_use]
    pub const fn is_reduce_only(self) -> bool {
        self.flags & Self::REDUCE_ONLY != 0
    }

    #[must_use]
    pub const fn expiry(self) -> CostExpiry {
        match self.flags & Self::EXPIRY {
            0 => CostExpiry::Combat,
            Self::END_OF_TURN => CostExpiry::Turn,
            Self::WHEN_PLAYED => CostExpiry::Played,
            _ => CostExpiry::TurnOrPlayed,
        }
    }

    const fn expiry_bits(self) -> u8 {
        self.flags & Self::EXPIRY
    }

    /// `LocalCostModifier.Modify`.
    #[must_use]
    pub fn modify(self, cur: i32) -> i32 {
        let amount = i32::from(self.amount);
        match (self.is_absolute(), self.is_reduce_only()) {
            (true, true) => cur.min(amount),
            (true, false) => amount,
            (false, true) => cur.min(cur + amount),
            (false, false) => cur + amount,
        }
    }

    /// A plain (not reduce-only) Relative modifier: these commute with each
    /// other, so two of them with the same expiry fold into one.
    const fn is_plain_relative(self) -> bool {
        self.flags & (Self::ABSOLUTE | Self::REDUCE_ONLY) == 0
    }

    /// A reduce-only Absolute (`min(cur, amount)`): these commute with each
    /// other, so two of them with the same expiry fold into one.
    const fn is_reduce_only_absolute(self) -> bool {
        self.flags & (Self::ABSOLUTE | Self::REDUCE_ONLY) == Self::ABSOLUTE | Self::REDUCE_ONLY
    }
}

impl fmt::Debug for LocalCostMod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match (self.is_absolute(), self.is_reduce_only()) {
            (true, true) => "min",
            (true, false) => "=",
            (false, true) => "+min",
            (false, false) => "+",
        };
        write!(f, "{kind}{}/{:?}", self.amount, self.expiry())
    }
}

/// C# `CardEnergyCost` (see the module docs).
///
/// `Copy` and 18 bytes. Equality and `Hash` see the base and the live
/// modifiers only; the unused tail is always zeroed. A card with no
/// modifiers hashes and prints exactly like the bare `i8` cost it replaced,
/// so search hashes and `Debug` strings of unmodified cards are unchanged.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CardEnergyCost {
    base: i8,
    len: u8,
    mods: [LocalCostMod; MAX_COST_MODS],
}

impl Default for CardEnergyCost {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Hash for CardEnergyCost {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.base.hash(h);
        if self.len != 0 {
            self.mods().hash(h);
        }
    }
}

impl fmt::Debug for CardEnergyCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len == 0 {
            return fmt::Debug::fmt(&self.base, f);
        }
        write!(f, "{}{:?}", self.base, self.mods())
    }
}

impl CardEnergyCost {
    /// A cost of `base` with no modifiers (C# `new CardEnergyCost(card,
    /// canonicalCost, costsX)`; this engine writes an X cost as `-1`).
    #[must_use]
    pub const fn new(base: i8) -> Self {
        Self {
            base,
            len: 0,
            mods: [LocalCostMod {
                amount: 0,
                flags: 0,
            }; MAX_COST_MODS],
        }
    }

    /// Rebuild a serialized cost: `base` and the modifier list exactly as
    /// [`Self::base`] and [`Self::mods`] returned them, with no compaction.
    /// `None` when the list is longer than [`MAX_COST_MODS`].
    #[must_use]
    pub fn from_raw(base: i8, mods: &[LocalCostMod]) -> Option<Self> {
        if mods.len() > MAX_COST_MODS {
            return None;
        }
        let mut out = Self::new(base);
        out.mods[..mods.len()].copy_from_slice(mods);
        out.len = u8::try_from(mods.len()).ok()?;
        Some(out)
    }

    /// C# `_base`: the cost before any local modifier (what
    /// `GetWithModifiers(CostModifiers.None)` returns).
    #[must_use]
    pub const fn base(&self) -> i8 {
        self.base
    }

    /// C# `SetCustomBaseCost`: replace the base and keep the modifiers. For
    /// card creation and the enchantments that rewrite a base
    /// (`TezcatarasEmber`), not for in-combat effects.
    pub fn set_base(&mut self, base: i8) {
        self.base = base;
    }

    /// The live modifiers, oldest first.
    #[must_use]
    pub fn mods(&self) -> &[LocalCostMod] {
        &self.mods[..usize::from(self.len)]
    }

    /// Whether the list holds `m` (tests and diagnostics: "was this card
    /// made free this turn?" is `has(LocalCostMod::FREE_THIS_TURN)`).
    #[must_use]
    pub fn has(&self, m: LocalCostMod) -> bool {
        self.mods().contains(&m)
    }

    /// C# `HasLocalModifiers`.
    #[must_use]
    pub const fn has_local_modifiers(&self) -> bool {
        self.len != 0
    }

    /// C# `GetWithModifiers(CostModifiers.Local)` BEFORE its final
    /// `Math.Max(0, num)`: the base folded through every modifier in order. A
    /// negative base (X cost, unplayable) is returned as is.
    #[must_use]
    pub fn local(&self) -> i32 {
        let base = i32::from(self.base);
        // Hot path: almost every card carries no modifier, so the read is
        // the base with no fold (X-cost / unplayable bases skip it too).
        if self.len == 0 || base < 0 {
            return base;
        }
        self.mods().iter().fold(base, |c, m| m.modify(c))
    }

    /// C# `SetUntilPlayed(cost, reduceOnly)`.
    pub fn set_until_played(&mut self, cost: i32, reduce_only: bool) {
        self.set(cost, CostExpiry::Played, reduce_only);
    }

    /// C# `SetThisTurnOrUntilPlayed(cost, reduceOnly)`; `SetToFreeThisTurn`
    /// is `SetThisTurnOrUntilPlayed(0)`.
    pub fn set_this_turn_or_until_played(&mut self, cost: i32, reduce_only: bool) {
        self.set(cost, CostExpiry::TurnOrPlayed, reduce_only);
    }

    /// C# `SetThisTurn(cost, reduceOnly)`.
    pub fn set_this_turn(&mut self, cost: i32, reduce_only: bool) {
        self.set(cost, CostExpiry::Turn, reduce_only);
    }

    /// C# `SetThisCombat(cost, reduceOnly)`; `SetToFreeThisCombat` is
    /// `SetThisCombat(0)`.
    pub fn set_this_combat(&mut self, cost: i32, reduce_only: bool) {
        self.set(cost, CostExpiry::Combat, reduce_only);
    }

    /// C# `AddUntilPlayed(amount)`.
    pub fn add_until_played(&mut self, amount: i32) {
        self.add(amount, CostExpiry::Played);
    }

    /// C# `AddThisTurn(amount)`.
    pub fn add_this_turn(&mut self, amount: i32) {
        self.add(amount, CostExpiry::Turn);
    }

    /// C# `AddThisCombat(amount)`.
    pub fn add_this_combat(&mut self, amount: i32) {
        self.add(amount, CostExpiry::Combat);
    }

    /// The `Set…` family: an Absolute modifier. C# skips `cost == 0` on a
    /// card whose canonical cost is negative; every modifier on a negative
    /// base is unobservable, so [`Self::push`] drops them all.
    pub fn set(&mut self, cost: i32, expiry: CostExpiry, reduce_only: bool) {
        self.push(LocalCostMod::absolute(clamp_i8(cost), expiry, reduce_only));
    }

    /// The `Add…` family: a Relative modifier, skipped when `amount` is 0
    /// (as C# does). No C# caller passes `reduceOnly` to an `Add…`.
    pub fn add(&mut self, amount: i32, expiry: CostExpiry) {
        if amount == 0 {
            return;
        }
        self.push(LocalCostMod::relative(clamp_i8(amount), expiry));
    }

    /// Append `m` (C# `_localModifiers.Add`), compacting the list with
    /// rewrites that leave every value it can ever produce unchanged:
    ///
    /// * A plain Absolute `m` hides every earlier modifier for as long as it
    ///   lives. An earlier modifier whose expiry bits include all of `m`'s
    ///   can never outlive it (any cleanup that drops `m` drops it too), so it
    ///   can never be read again and is removed. A `…ThisCombat` Absolute has
    ///   no expiry bit, so it clears the list; a Snecko roll replaces the
    ///   previous turn-or-played modifiers; Flatten's repeated `SetThisTurn(0)`
    ///   keeps one.
    /// * Plain Relatives commute with each other (they only add), so `m` folds
    ///   into an earlier plain Relative with the same expiry when only plain
    ///   Relatives lie between them: both always expire together. Kingly
    ///   Kick's per-draw `-1` and Stomp's per-Attack `-1` stay one modifier.
    /// * Reduce-only Absolutes (`min(cur, n)`) commute with each other the
    ///   same way, and `UpgradeBy`'s clamp preserves their minimum.
    ///
    /// After compaction a card holds one entry per alternation of modifier
    /// kinds and expiries, a handful at most. If a pathological sequence
    /// still fills all [`MAX_COST_MODS`] slots, the oldest modifier is
    /// dropped: it is the one most likely to be hidden by a later Absolute.
    /// Debug builds assert that this never happens.
    pub fn push(&mut self, m: LocalCostMod) {
        if self.base < 0 {
            return;
        }
        if m.is_absolute() && !m.is_reduce_only() {
            let exp = m.expiry_bits();
            self.retain(|p| p.expiry_bits() & exp != exp);
        } else if let Some(j) = self.fold_target(m) {
            let slot = &mut self.mods[j];
            if m.is_plain_relative() {
                slot.amount = slot.amount.saturating_add(m.amount);
                if slot.amount == 0 {
                    self.remove(j);
                }
            } else {
                slot.amount = slot.amount.min(m.amount);
            }
            return;
        }
        if usize::from(self.len) == MAX_COST_MODS {
            debug_assert!(
                false,
                "card energy cost overflowed {MAX_COST_MODS} modifiers"
            );
            self.remove(0);
        }
        self.mods[usize::from(self.len)] = m;
        self.len += 1;
    }

    /// The earlier modifier `m` folds into (see [`Self::push`]), if any.
    fn fold_target(&self, m: LocalCostMod) -> Option<usize> {
        let same_kind: fn(LocalCostMod) -> bool = if m.is_plain_relative() {
            LocalCostMod::is_plain_relative
        } else if m.is_reduce_only_absolute() {
            LocalCostMod::is_reduce_only_absolute
        } else {
            return None;
        };
        for (j, p) in self.mods().iter().enumerate().rev() {
            if !same_kind(*p) {
                return None;
            }
            if p.expiry_bits() == m.expiry_bits() {
                return Some(j);
            }
        }
        None
    }

    fn remove(&mut self, j: usize) {
        let len = usize::from(self.len);
        self.mods.copy_within(j + 1..len, j);
        self.mods[len - 1] = LocalCostMod::default();
        self.len -= 1;
    }

    /// Keep the modifiers `keep` accepts, in order; `true` if any was
    /// dropped. Leaves the unused tail zeroed.
    fn retain(&mut self, keep: impl Fn(LocalCostMod) -> bool) -> bool {
        let len = usize::from(self.len);
        let mut w = 0;
        for r in 0..len {
            let m = self.mods[r];
            if keep(m) {
                self.mods[w] = m;
                w += 1;
            }
        }
        for slot in &mut self.mods[w..len] {
            *slot = LocalCostMod::default();
        }
        self.len = w as u8;
        w != len
    }

    /// C# `EndOfTurnCleanup`: drop the `EndOfTurn` modifiers. `true` if any
    /// was dropped.
    pub fn end_of_turn_cleanup(&mut self) -> bool {
        self.len != 0 && self.retain(|m| m.flags & LocalCostMod::END_OF_TURN == 0)
    }

    /// C# `AfterCardPlayedCleanup`: drop the `WhenPlayed` modifiers. `true`
    /// if any was dropped.
    pub fn after_card_played_cleanup(&mut self) -> bool {
        self.len != 0 && self.retain(|m| m.flags & LocalCostMod::WHEN_PLAYED == 0)
    }

    /// Drop every modifier (the combat is over: C# combat cards are
    /// discarded with theirs).
    pub fn clear_modifiers(&mut self) {
        self.retain(|_| false);
    }

    /// C# `UpgradeBy(addend)`: move the base (floored at 0) and, when that
    /// lowers it, clamp every Absolute amount above the new base down to it.
    /// A negative base (X cost) is left alone, as C# skips `CostsX`.
    pub fn upgrade_by(&mut self, addend: i32) {
        if addend == 0 || self.base < 0 {
            return;
        }
        let old = i32::from(self.base);
        let new = (old + addend).max(0);
        let new8 = clamp_i8(new);
        if new < old {
            let len = usize::from(self.len);
            for m in &mut self.mods[..len] {
                if m.is_absolute() && m.amount > new8 {
                    m.amount = new8;
                }
            }
        }
        self.base = new8;
    }

    /// C# `ResetForDowngrade`: the base returns to `canonical`; the
    /// modifiers stay.
    pub fn reset_for_downgrade(&mut self, canonical: i8) {
        self.base = canonical;
    }
}

fn clamp_i8(v: i32) -> i8 {
    i8::try_from(v.clamp(i32::from(i8::MIN), i32::from(i8::MAX))).unwrap_or(0)
}

// 1 (base) + 1 (len) + 8 × 2 (mods).
const _: () = assert!(core::mem::size_of::<CardEnergyCost>() == 18);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_folds_in_order_and_skips_negative_bases() {
        let mut c = CardEnergyCost::new(3);
        c.add_this_combat(-1);
        c.set_this_turn_or_until_played(0, false);
        c.add_this_combat(-1);
        assert_eq!(c.local(), -1);
        assert!(c.end_of_turn_cleanup());
        assert_eq!(c.local(), 1);
        let mut x = CardEnergyCost::new(-1);
        x.set_this_combat(0, false);
        assert_eq!(x.local(), -1);
        assert!(!x.has_local_modifiers());
    }

    #[test]
    fn compaction_keeps_the_list_short() {
        let mut c = CardEnergyCost::new(4);
        for _ in 0..20 {
            c.add_this_combat(-1);
        }
        assert_eq!(c.mods().len(), 1);
        for _ in 0..5 {
            c.set_this_turn(0, false);
        }
        assert_eq!(c.mods().len(), 2);
        c.set_this_combat(2, false);
        assert_eq!(c.mods().len(), 1);
        assert_eq!(c.local(), 2);
        c.add_this_combat(1);
        c.add_this_combat(-1);
        assert_eq!(c.mods().len(), 1, "a zero sum drops out");
    }

    #[test]
    fn no_modifier_hash_and_debug_match_the_bare_base() {
        use std::collections::hash_map::DefaultHasher;
        let h = |v: &dyn Fn(&mut DefaultHasher)| {
            let mut s = DefaultHasher::new();
            v(&mut s);
            s.finish()
        };
        let c = CardEnergyCost::new(3);
        assert_eq!(h(&|s| c.hash(s)), h(&|s| 3i8.hash(s)));
        assert_eq!(format!("{c:?}"), "3");
    }

    #[test]
    fn upgrade_clamps_absolutes_above_a_lowered_base() {
        let mut c = CardEnergyCost::new(3);
        c.set_this_combat(2, false);
        c.add_this_combat(1);
        c.upgrade_by(-2);
        assert_eq!(c.base(), 1);
        assert_eq!(c.local(), 2, "clamped 1 + 1");
        c.reset_for_downgrade(3);
        assert_eq!(c.local(), 2);
    }
}
