//! [`InlineKv`] — a fixed-capacity, `Copy` key→i32 map for [`crate::Card`].
//!
//! Replaces the two `SmallVec<[(K, i32); N]>` fields (`combat_vars`,
//! `enchantments`) that were the only thing keeping `Card` from being
//! `Copy`. With `Card: Copy`, `Vec<Card>::clone` — the per-simulation
//! card-arena deep-clone in MCTS (2–3 full `CombatState` clones per
//! simulation) — specializes to a single `memcpy` instead of an
//! element-wise loop with per-element branch + drop glue. See
//! `docs/perf/2026-07-05-mcts-single-thread-speedup.md` ("Remaining
//! levers": the clone complex, with `Card: Copy` named as the untested
//! remaining option).
//!
//! # Behavior contract (load-bearing)
//!
//! This type must be observationally IDENTICAL to the `SmallVec` it
//! replaces, because `Card`'s derived `PartialEq` and its `Debug` output
//! both feed search behavior:
//!
//! * `PartialEq` participates in [`crate::cards::cards_observationally_equal`],
//!   which feeds the DrawChance segment dedup and (via
//!   `crate::hashing::card_observational_key`, which hashes
//!   `format!("{card:?}")`) the transposition/dedup hashes. Equality is
//!   therefore **insertion-order-sensitive over the live entries** —
//!   exactly `SmallVec`'s slice equality. Canonical zeroed tail slots
//!   (see below) make the whole-array compare equivalent to a live-
//!   prefix compare.
//! * `Debug` must format **byte-identically** to `SmallVec`'s (a plain
//!   slice-style list of the live entries), so the Debug-string hashes
//!   are unchanged. The manual impl below prints only the live prefix.
//!
//! # Tail canonicalization invariant
//!
//! Slots at index `>= len` are ALWAYS `(K::FILL, 0)`. Every constructor
//! and mutator preserves this ([`InlineKv::new`] fills, [`InlineKv::push`]
//! writes at `len`, [`InlineKv::clear`] re-fills, and the in-place writers
//! [`InlineKv::set`] / [`InlineKv::add`] / [`InlineKv::zero`] only touch the
//! live prefix). This is what lets `PartialEq`/`Eq` compare the whole fixed
//! array without comparing garbage.
//!
//! # "May be nonzero" mask (J02)
//!
//! `nz` is a 24-bit derived view: bit [`InlineKvKey::nz_index`] is set
//! whenever a nonzero value is written under that key, and only
//! [`InlineKv::clear`] resets it. So a clear bit PROVES every live entry for
//! the key holds 0 (a superset filter: writing 0 keeps the bit, and keys may
//! share a bit). [`InlineKv::get`] answers 0 from the bit alone, and the
//! per-turn card sweeps skip cards whose bit is clear. The invariant is
//! enforced by the type: the fields are private and every value writer is a
//! method here (there is deliberately no `iter_mut`). `nz` lives in what was
//! tail padding after `len`, so the struct size is unchanged, and it is left
//! out of `PartialEq`, `Hash` and `Debug` (it depends on write history, not
//! only on content).
//!
//! # Capacity overflow
//!
//! [`InlineKv::push`] PANICS (loudly, naming the key and dumping the live
//! entries) when the fixed capacity is exceeded — a silent drop would be a
//! correctness bug, and spilling to the heap would forfeit `Copy`.
//! Capacities on the `Card` aliases are sized against the worst DISTINCT-KEY
//! set a card can legitimately accumulate, not the typical stack — slots are
//! never reclaimed, so "realistic" undersizes them (GH #474 hit exactly
//! that; see the alias docs in `cards.rs`). The whole validation suite
//! (nextest, parity, fuzz, full random-run sweeps) runs against these caps,
//! so an overflow shows up as a panic at the offending site, never as
//! silent state loss.
//!
//! [`InlineKv`]: crate::inline_kv::InlineKv
//! [`InlineKv::new`]: crate::inline_kv::InlineKv::new
//! [`InlineKv::push`]: crate::inline_kv::InlineKv::push
//! [`InlineKv::clear`]: crate::inline_kv::InlineKv::clear
//! [`InlineKv::set`]: crate::inline_kv::InlineKv::set
//! [`InlineKv::add`]: crate::inline_kv::InlineKv::add
//! [`InlineKv::zero`]: crate::inline_kv::InlineKv::zero
//! [`InlineKvKey::nz_index`]: crate::inline_kv::InlineKvKey::nz_index
//! [`InlineKv::get`]: crate::inline_kv::InlineKv::get

use core::fmt;

/// Key trait for [`InlineKv`]: fieldless `Copy` enums with a designated
/// fill variant for canonical tail slots. The fill variant is arbitrary
/// (any variant works — tails are also length-masked); by convention the
/// first declared variant.
pub trait InlineKvKey: Copy + Eq + fmt::Debug {
    /// Variant used to fill unused tail slots (paired with amount `0`).
    const FILL: Self;

    /// This key's bit in the derived "may be nonzero" mask; must be
    /// `< NZ_BITS`. Keys may share a bit (the mask is a superset filter, so
    /// aliasing only costs a false "maybe").
    fn nz_index(self) -> usize;
}

/// Width of [`InlineKv`]'s "may be nonzero" mask: the three bytes of tail
/// padding after `len`.
pub const NZ_BITS: usize = 24;

/// Fixed-capacity insertion-ordered `(K, i32)` map. `Copy`. See the
/// module docs for the equality / Debug / tail-canonicalization contract.
///
/// Keys and values live in two parallel arrays rather than one array of
/// `(K, i32)` pairs: a one-byte key paired with an `i32` pads every pair to
/// 8 bytes, so the split layout stores a 16-entry map in 84 bytes instead of
/// 132. `Card` holds two of these and the card arena is memcpy-cloned on
/// every MCTS simulation.
#[derive(Clone, Copy)]
pub struct InlineKv<K: InlineKvKey, const N: usize> {
    len: u8,
    /// Derived "may be nonzero" mask (see the module docs); excluded from
    /// `PartialEq`, `Hash` and `Debug`.
    nz: [u8; 3],
    keys: [K; N],
    vals: [i32; N],
}

/// The live entries of an [`InlineKv`], by value, in insertion order.
pub type InlineKvIter<'a, K> = core::iter::Zip<
    core::iter::Copied<core::slice::Iter<'a, K>>,
    core::iter::Copied<core::slice::Iter<'a, i32>>,
>;

/// Equality over the live content only (`len` + the canonical arrays),
/// exactly the derived comparison before `nz` existed.
impl<K: InlineKvKey, const N: usize> PartialEq for InlineKv<K, N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.keys == other.keys && self.vals == other.vals
    }
}
impl<K: InlineKvKey, const N: usize> Eq for InlineKv<K, N> {}

impl<K: InlineKvKey, const N: usize> InlineKv<K, N> {
    /// Empty map (all slots canonical `(K::FILL, 0)`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            len: 0,
            nz: [0; 3],
            keys: [K::FILL; N],
            vals: [0; N],
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Live keys, insertion-ordered (the tail is not exposed).
    #[must_use]
    pub fn keys(&self) -> &[K] {
        &self.keys[..self.len as usize]
    }

    /// Live values, in the order of [`Self::keys`].
    #[must_use]
    pub fn values(&self) -> &[i32] {
        &self.vals[..self.len as usize]
    }

    /// Live entries, insertion-ordered, by value.
    pub fn iter(&self) -> InlineKvIter<'_, K> {
        self.keys()
            .iter()
            .copied()
            .zip(self.values().iter().copied())
    }

    /// Live entries as a `Vec` (tests and diagnostics; allocates).
    #[must_use]
    pub fn to_vec(&self) -> Vec<(K, i32)> {
        self.iter().collect()
    }

    #[must_use]
    pub fn first(&self) -> Option<(K, i32)> {
        self.iter().next()
    }

    /// Append an entry. Mirrors `SmallVec::push` for the live prefix.
    ///
    /// # Panics
    /// When the fixed capacity `N` is exhausted. See the module docs —
    /// this is the deliberate overflow semantic, not a recoverable error.
    pub fn push(&mut self, kv: (K, i32)) {
        let i = self.len as usize;
        assert!(
            i < N,
            "InlineKv<{}, {N}> capacity exceeded pushing {:?} — raise the \
             capacity on the Card field alias (cards.rs) if this stacking is \
             legitimate; live entries: {:?}",
            core::any::type_name::<K>(),
            kv.0,
            self.to_vec(),
        );
        self.keys[i] = kv.0;
        self.vals[i] = kv.1;
        self.len += 1;
        self.note_value(kv.0, kv.1);
    }

    /// Remove all entries (restores every slot to the canonical tail and
    /// clears the "may be nonzero" mask).
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// `false` proves every live entry for `key` holds 0 (or there is none);
    /// `true` means one may be nonzero. One bit test.
    ///
    /// # Panics
    /// In debug builds, if it answers `false` while a live entry for `key`
    /// is nonzero (the superset invariant broke).
    #[inline]
    #[must_use]
    pub fn maybe_nonzero(&self, key: K) -> bool {
        let i = key.nz_index();
        debug_assert!(i < NZ_BITS, "nz_index {i} of {key:?} out of range");
        let maybe = self.nz[i / 8] & (1 << (i % 8)) != 0;
        debug_assert!(
            maybe || self.iter().all(|(k, v)| k != key || v == 0),
            "InlineKv mask says {key:?} is zero but the entries are {:?}",
            self.to_vec()
        );
        maybe
    }

    /// Value of the FIRST live entry for `key`, or 0 when there is none.
    /// Answered from the mask alone when `key`'s bit is clear (every entry
    /// for it is then 0).
    #[inline]
    #[must_use]
    pub fn get(&self, key: K) -> i32 {
        if !self.maybe_nonzero(key) {
            return 0;
        }
        self.iter()
            .find_map(|(k, v)| (k == key).then_some(v))
            .unwrap_or(0)
    }

    /// Overwrite the first live entry for `key` with `value`, or append
    /// `(key, value)` when there is none.
    ///
    /// # Panics
    /// When appending past the fixed capacity (see [`Self::push`]).
    #[inline]
    pub fn set(&mut self, key: K, value: i32) {
        if let Some(slot) = self.first_slot_mut(key) {
            *slot = value;
            self.note_value(key, value);
        } else {
            self.push((key, value));
        }
    }

    /// Add `delta` (saturating) to the first live entry for `key`, or append
    /// `(key, delta)` when there is none.
    ///
    /// # Panics
    /// When appending past the fixed capacity (see [`Self::push`]).
    #[inline]
    pub fn add(&mut self, key: K, delta: i32) {
        if let Some(slot) = self.first_slot_mut(key) {
            *slot = slot.saturating_add(delta);
            let value = *slot;
            self.note_value(key, value);
        } else {
            self.push((key, delta));
        }
    }

    /// Write 0 over the first live entry for `key`, if any. Never appends.
    /// Writing 0 keeps the mask a superset, so it is left as is.
    #[inline]
    pub fn zero(&mut self, key: K) {
        if let Some(slot) = self.first_slot_mut(key) {
            *slot = 0;
        }
    }

    #[inline]
    fn first_slot_mut(&mut self, key: K) -> Option<&mut i32> {
        let len = self.len as usize;
        let i = self.keys[..len].iter().position(|k| *k == key)?;
        Some(&mut self.vals[i])
    }

    /// Record a write of `value` under `key` in the mask.
    #[inline]
    fn note_value(&mut self, key: K, value: i32) {
        if value != 0 {
            let i = key.nz_index();
            debug_assert!(i < NZ_BITS, "nz_index {i} of {key:?} out of range");
            self.nz[i / 8] |= 1 << (i % 8);
        }
    }
}

impl<K: InlineKvKey, const N: usize> Default for InlineKv<K, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Slice-style Debug over the LIVE entries only — must stay byte-identical
/// to the replaced `SmallVec`'s Debug output (it was hashed via
/// `format!("{card:?}")` in `hashing::card_observational_key` before
/// PR #469 moved that key to a derived field hash; keeping the format
/// identical preserves every remaining Debug-encoding consumer).
impl<K: InlineKvKey, const N: usize> fmt::Debug for InlineKv<K, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Slice-style Hash over the LIVE entries only — byte-identical hash
/// stream to the replaced `SmallVec`'s `Hash` (which delegates to the
/// slice: length prefix + each element). `Card` derives `Hash` and feeds
/// `hashing::card_observational_key` (PR #469's field-hash observational
/// key), so hashing the canonical tail slots — or even just a different
/// length encoding — would silently change every observational key.
impl<K: InlineKvKey + core::hash::Hash, const N: usize> core::hash::Hash for InlineKv<K, N> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        // `<[(K, i32)]>::hash`: the length prefix, then each pair's fields.
        state.write_usize(self.len());
        for (k, v) in self {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl<K: InlineKvKey, const N: usize> FromIterator<(K, i32)> for InlineKv<K, N> {
    fn from_iter<I: IntoIterator<Item = (K, i32)>>(iter: I) -> Self {
        let mut out = Self::new();
        for kv in iter {
            out.push(kv);
        }
        out
    }
}

impl<'a, K: InlineKvKey, const N: usize> IntoIterator for &'a InlineKv<K, N> {
    type Item = (K, i32);
    type IntoIter = InlineKvIter<'a, K>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
    enum K {
        A,
        B,
        C,
    }
    impl InlineKvKey for K {
        const FILL: Self = K::A;
        fn nz_index(self) -> usize {
            self as usize
        }
    }

    type M = InlineKv<K, 3>;

    #[test]
    fn push_iter_first_len() {
        let mut m = M::new();
        assert!(m.is_empty());
        m.push((K::B, 2));
        m.push((K::C, -1));
        assert_eq!(m.len(), 2);
        assert_eq!(m.first(), Some((K::B, 2)));
        assert_eq!(m.to_vec(), &[(K::B, 2), (K::C, -1)]);
        assert_eq!(m.iter().count(), 2);
    }

    /// The split key/value layout hashes exactly as the `[(K, i32)]` slice
    /// it replaced (the search hashes and the streams seeded from them).
    #[test]
    fn hash_matches_the_pair_slice() {
        use core::hash::{Hash, Hasher};
        fn h(x: &(impl Hash + ?Sized)) -> u64 {
            let mut s = std::collections::hash_map::DefaultHasher::new();
            x.hash(&mut s);
            s.finish()
        }
        for m in [
            M::new(),
            [(K::B, 2)].into_iter().collect(),
            [(K::C, -1), (K::A, 7), (K::B, 0)].into_iter().collect(),
        ] {
            assert_eq!(h(&m), h(m.to_vec().as_slice()));
            assert_eq!(format!("{m:?}"), format!("{:?}", m.to_vec()));
        }
    }

    #[test]
    fn eq_is_order_sensitive_over_live_prefix_like_smallvec() {
        let ab: M = [(K::A, 1), (K::B, 2)].into_iter().collect();
        let ba: M = [(K::B, 2), (K::A, 1)].into_iter().collect();
        let ab2: M = [(K::A, 1), (K::B, 2)].into_iter().collect();
        assert_eq!(ab, ab2);
        assert_ne!(ab, ba); // insertion order participates, as with SmallVec
                            // A live (FILL, 0) entry is NOT equal to an empty map (len differs)
        let fill0: M = [(K::A, 0)].into_iter().collect();
        assert_ne!(fill0, M::new());
    }

    #[test]
    fn set_updates_in_place_and_preserves_eq() {
        let mut m: M = [(K::B, 1)].into_iter().collect();
        m.set(K::B, 7);
        let want: M = [(K::B, 7)].into_iter().collect();
        assert_eq!(m, want);
        // Absent key: appended at the end, like the old find-or-push.
        m.set(K::C, 2);
        assert_eq!(m.to_vec(), &[(K::B, 7), (K::C, 2)]);
    }

    #[test]
    fn add_saturates_and_zero_never_appends() {
        let mut m = M::new();
        m.add(K::B, 3);
        m.add(K::B, i32::MAX);
        assert_eq!(m.to_vec(), &[(K::B, i32::MAX)]);
        m.zero(K::C);
        assert_eq!(m.len(), 1, "zero must not append an absent key");
        m.zero(K::B);
        assert_eq!(m.to_vec(), &[(K::B, 0)]);
    }

    /// The mask is history, not content: two maps with the same live entries
    /// but different write histories are equal, hash alike and print alike,
    /// and `get` answers the same on both.
    #[test]
    fn nz_mask_is_invisible_to_eq_hash_debug() {
        use core::hash::{Hash, Hasher};
        fn h(x: &impl Hash) -> u64 {
            let mut s = std::collections::hash_map::DefaultHasher::new();
            x.hash(&mut s);
            s.finish()
        }
        let mut a = M::new();
        a.set(K::B, 5);
        a.set(K::B, 0);
        let b: M = [(K::B, 0)].into_iter().collect();
        assert!(a.maybe_nonzero(K::B));
        assert!(!b.maybe_nonzero(K::B));
        assert_eq!(a, b);
        assert_eq!(h(&a), h(&b));
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        assert_eq!(a.get(K::B), 0);
        assert_eq!(b.get(K::B), 0);
        a.clear();
        assert!(!a.maybe_nonzero(K::B), "clear resets the mask");
    }

    #[test]
    fn clear_restores_canonical_tail_so_eq_holds() {
        let mut m: M = [(K::C, 5), (K::B, 6)].into_iter().collect();
        m.clear();
        assert_eq!(m, M::new());
        // Rebuild after clear must equal a fresh build (tail canonical).
        m.push((K::C, 5));
        let fresh: M = [(K::C, 5)].into_iter().collect();
        assert_eq!(m, fresh);
    }

    #[test]
    fn debug_matches_smallvec_slice_format() {
        let m: M = [(K::B, 2), (K::C, -1)].into_iter().collect();
        // SmallVec's Debug delegates to the slice: `[(B, 2), (C, -1)]`.
        assert_eq!(format!("{m:?}"), "[(B, 2), (C, -1)]");
        assert_eq!(format!("{:?}", M::new()), "[]");
        // Alternate form must also match the slice formatter (the
        // observational-key hash uses the non-alternate form, but keep
        // parity everywhere).
        let sv: &[(K, i32)] = &[(K::B, 2), (K::C, -1)];
        assert_eq!(format!("{m:#?}"), format!("{sv:#?}"));
    }

    #[test]
    fn hash_matches_slice_hash_and_ignores_tail() {
        use core::hash::{Hash, Hasher};
        fn h(x: &impl Hash) -> u64 {
            let mut s = std::collections::hash_map::DefaultHasher::new();
            x.hash(&mut s);
            s.finish()
        }
        let m: M = [(K::B, 2), (K::C, -1)].into_iter().collect();
        let sv: &[(K, i32)] = &[(K::B, 2), (K::C, -1)];
        // Byte-identical hash stream to the slice (== SmallVec's Hash).
        assert_eq!(h(&m), h(&sv));
        // A cleared-and-refilled map hashes like a fresh one (tail never
        // participates).
        let mut n: M = [(K::A, 9), (K::B, 8), (K::C, 7)].into_iter().collect();
        n.clear();
        n.push((K::B, 2));
        n.push((K::C, -1));
        assert_eq!(h(&n), h(&m));
    }

    #[test]
    #[should_panic(expected = "capacity exceeded")]
    fn push_past_capacity_panics_loudly() {
        let mut m = M::new();
        m.push((K::A, 1));
        m.push((K::B, 2));
        m.push((K::C, 3));
        m.push((K::A, 4));
    }
}
