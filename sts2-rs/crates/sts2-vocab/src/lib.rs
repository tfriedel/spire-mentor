//! PPO id vocabularies — Rust port of `sts2-pycli/python/ppo/vocab.py`.
//!
//! Lives in its own crate (moved out of `sts2_runai::ppo::vocab`, which
//! re-exports it) so `sts2-wire` and the browser workbench can name the
//! registries without pulling tract-onnx / rayon into a wasm build.
//!
//! Original header follows.
//!
//! Rust port of `sts2-pycli/python/ppo/vocab.py`
//! (porting authority; do not consult the Python sim for this).
//!
//! Card/relic/potion/event/boss universes are canonical `UPPER_SNAKE` names
//! sourced from the same `sts2_core` enums that back
//! `sts2-py::state_dict::registry_names_for` — this module is now the
//! single source and `state_dict.rs` delegates its `card`/`potion`/
//! `event`/`relic_id`/`boss` arms here (Task 4, 2026-07-07
//! ppo-rust-inference plan). Index `len(names)` is a reserved OOV/pad slot,
//! so every [`Registry`] is built with `names.len() + 1` rows.
//!
//! Their ORDER comes from the append-only pin files in `vocab_pins/`, not
//! from a sort: a name's position is its embedding row, so sorting made
//! every content addition renumber the rows a deployed champion had already
//! learned (GH #544). See [`registry_names`].
//!
//! The relic universe is `"relic_id"` (the run-side `RelicId` enum), NOT
//! `"relic"` (the ~111-entry combat `RelicKind` subset): Neow blessings,
//! event relics and shop stock exist only in `RelicId`.
//!
//! Normalization wrinkle: `card_reward_option_ids` returns
//! `"CARD.ULTIMATE_STRIKE"` while `master_deck_detailed` returns
//! PascalCase (`"UltimateStrike"`). Both must map to the canonical
//! `"ULTIMATE_STRIKE"` via [`canonical`].

/// Content hash of THIS crate's build inputs, stamped at build time by
/// `build.rs`. Compiled INTO the crate so a stale `sts2-vocab` rlib carries its
/// old hash; one of the parts of `sts2_rs.src_hash()`. See
/// `sts2_core::SRC_HASH`.
pub const SRC_HASH: &str = env!("STS2_VOCAB_SRC_HASH");

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use sts2_core::cards::CardId;
use sts2_core::potions::PotionKind;
use sts2_core::{EventId, GameVersion, RelicId};

/// Fixed map room-type vocab (sts2 `MapPointType`, snake form from
/// `act_map()`). Order is embedding-index-significant — mirrors
/// `vocab.py::ROOM_TYPES` verbatim.
pub const ROOM_TYPES: [&str; 9] = [
    "unassigned",
    "unknown",
    "monster",
    "elite",
    "boss",
    "shop",
    "rest_site",
    "treasure",
    "ancient",
];

/// `ROOM_TYPES.len() + 1` (reserved OOV row).
pub const N_ROOM: usize = ROOM_TYPES.len() + 1;

/// Look up a room type's embedding index, falling back to the reserved
/// OOV row (`ROOM_TYPES.len()`) for anything unrecognized.
#[must_use]
pub fn room_index(room: &str) -> usize {
    ROOM_TYPES
        .iter()
        .position(|&r| r == room)
        .unwrap_or(ROOM_TYPES.len())
}

/// Atomic (non-card/relic/potion/path/event) action kinds — mirrors
/// `vocab.py::FIXED_KINDS` verbatim (order is embedding-index-significant).
pub const FIXED_KINDS: [&str; 12] = [
    "none",
    "skip_card",
    "take_relic",
    "skip_relic",
    "leave_shop",
    "shop_remove",
    "rest_heal",
    "rest_smith",
    "rest_other",
    "open_chest",
    "sacrifice",
    // Appended (not inserted) so every pre-existing embedding row keeps its
    // index and `model.load_compat` can warm-start an older checkpoint.
    "decline_potion",
];

/// The PAD index for `cat[6]`, mirroring `vocab.py::N_FIXED`.
///
/// `len()`, NOT `len() + 1`: the net allocates `N_FIXED + 1` rows and pins
/// `padding_idx` to the last, so the table always carried one SPARE row
/// above the named entries. GH #379's `"decline_potion"` consumes that
/// spare, so the pad index does not move and the table does not grow —
/// which is what keeps every exported `.onnx` champion loadable (its
/// embedding tables are shape-FROZEN). The next entry after it has no
/// spare left and WILL require re-exporting every champion.
pub const N_FIXED: usize = FIXED_KINDS.len();

/// Look up a fixed-kind embedding index, falling back to `"none"`'s
/// index for anything unrecognized (mirrors `vocab.py::fixed_index`).
#[must_use]
pub fn fixed_index(kind: &str) -> usize {
    FIXED_KINDS
        .iter()
        .position(|&k| k == kind)
        .unwrap_or(0 /* "none" */)
}

/// Decision-surface vocab for the state token — mirrors
/// `vocab.py::SURFACES` verbatim (order is embedding-index-significant).
pub const SURFACES: [&str; 16] = [
    "path",
    "card-reward",
    "card-remove",
    "card-upgrade",
    "card-transform",
    "card-enchant",
    "card-grant",
    "card-smith",
    "boss-relic",
    "relic-reward",
    "shop",
    "event",
    "rest",
    "treasure",
    "unknown",
    // Appended AFTER "unknown" on purpose: `surface_index` now resolves the
    // fallback BY NAME, so the tail is free real estate and rows 0..14 keep
    // the indices every shipped champion was trained with.
    "potion-swap",
];

/// The PAD index for `cat[8]`, mirroring `vocab.py::N_SURFACE`.
/// `len()`, NOT `len() + 1` — same spare-row argument as [`N_FIXED`].
pub const N_SURFACE: usize = SURFACES.len();

/// Look up a decision-surface embedding index, falling back to
/// `"unknown"`'s index for anything unrecognized.
#[must_use]
pub fn surface_index(s: &str) -> usize {
    // By NAME, not `len() - 1`: "unknown" stopped being the last entry when
    // "potion-swap" was appended (appending is what keeps older checkpoints
    // loadable), and a positional fallback would have silently started
    // resolving every unrecognized surface to "potion-swap".
    let unknown = SURFACES
        .iter()
        .position(|&x| x == "unknown")
        .expect("SURFACES must carry an \"unknown\" row");
    SURFACES.iter().position(|&x| x == s).unwrap_or(unknown)
}

/// `encoder.py::_KIND_SURFACE` verbatim: an action kind's surface-defining
/// label, or `None` for kinds that never define a surface (`unknown`
/// included: it carries no entry in the Python dict either). Every label is
/// a [`SURFACES`] row.
#[must_use]
pub fn kind_surface(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "move" => "path",
        "pick_card_reward" | "sacrifice_card_reward" => "card-reward",
        "pick_relic_reward" | "skip_relic" => "relic-reward",
        "pick_boss_relic" => "boss-relic",
        "smith_card" => "card-smith",
        "remove_deck_card" => "card-remove",
        "upgrade_deck_card" => "card-upgrade",
        "transform_deck_card" | "confirm_deck_selection" => "card-transform",
        "enchant_deck_card" => "card-enchant",
        "grant_deck_card" => "card-grant",
        "shop_buy_card" | "shop_buy_relic" | "shop_buy_potion" | "shop_remove_card"
        | "leave_shop" => "shop",
        "rest_option" => "rest",
        "event_choice" => "event",
        "collect_treasure" => "treasure",
        "potion_swap" => "potion-swap",
        _ => return None,
    })
}

/// `encoder.py::fine_surface` over the decision's action kinds, in order:
/// the first surface-defining kind, skipping the deferring `skip_relic` /
/// `leave_shop` on a first pass; any on a second; else `"unknown"`.
#[must_use]
pub fn fine_surface<'a, I>(kinds: I) -> &'static str
where
    I: IntoIterator<Item = &'a str>,
    I::IntoIter: Clone,
{
    const DEFERRING: [&str; 2] = ["skip_relic", "leave_shop"];
    let kinds = kinds.into_iter();
    kinds
        .clone()
        .filter(|k| !DEFERRING.contains(k))
        .find_map(kind_surface)
        .or_else(|| kinds.clone().find_map(kind_surface))
        .unwrap_or("unknown")
}

/// Which deck operation a CARD-section action token belongs to (GH #545) —
/// mirrors `vocab.py::DECK_OPS` verbatim (order is embedding-index-significant).
///
/// WHY THIS IS ITS OWN SLOT AND NOT A `FIXED_KINDS` ENTRY. `shop_remove_card`
/// separates itself with `cat[6] = fixed_index("shop_remove")`, and copying
/// that for the other ops is the obvious fix — but `N_FIXED` has no spare row
/// left (see its comment above), so five more entries would move `pad[6]`
/// from 12 to 17 and every committed champion's fixed table would stop
/// loading at once. An appended slot costs nothing instead: both consumers
/// read the `cat` width off the graph they loaded, exactly as they do for
/// `emb_boss`.
pub const DECK_OPS: [&str; 11] = [
    "none",
    "reward",
    "smith",
    "remove",
    "upgrade",
    "transform",
    "enchant",
    "grant",
    "shop_buy",
    "shop_remove",
    // A Scroll Boxes pack (GH #978, `card_bundles`). Took the spare row.
    "bundle",
];

/// The PAD index for `cat[10]`, mirroring `vocab.py::N_DECK_OP`, pinned at 11.
///
/// The table was defined with one SPARE row so the next deck op was a free
/// append; `"bundle"` (GH #978) took it, so this is now `len()`. The NEXT op
/// moves the pad and every committed champion's `cat[10]` with it (the
/// `N_FIXED` trap, GH #545): widen the table and re-export the fleet first.
pub const N_DECK_OP: usize = 11;
const _: () = assert!(N_DECK_OP == DECK_OPS.len());

/// Row for a deck-op name; unknown names fall back to `"none"` (row 0).
#[must_use]
pub fn deck_op_index(kind: &str) -> usize {
    DECK_OPS.iter().position(|&x| x == kind).unwrap_or(0)
}

/// Event option-index embedding rows (options beyond this clamp to the
/// last row).
pub const MAX_OPT_IDX: usize = 8;

/// Max upgrade count an embedding slot is allocated for (Searing Blow
/// etc. can exceed normal +1; clamp into this range).
pub const MAX_UPGRADE: i64 = 16;

/// `"CARD.ULTIMATE_STRIKE"` / `"UltimateStrike"` → `"ULTIMATE_STRIKE"`.
///
/// Ports `vocab.py::canonical`: strip any `PREFIX.` up to (and
/// including) the first `.`, then insert `_` at camel-case boundaries,
/// then uppercase. The Python side uses a two-lookaround regex
/// (`(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])`); the `regex`
/// crate has no lookaround support, so this is a char scan with the
/// equivalent boundary rule: insert `_` before an uppercase char when
/// (prev is lowercase/digit) OR (prev is uppercase AND next is
/// lowercase).
#[must_use]
pub fn canonical(name: &str) -> String {
    let s = match name.split_once('.') {
        Some((_, rest)) => rest,
        None => name,
    };
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len() + 4);
    for (i, &c) in chars.iter().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            let prev = chars[i - 1];
            let boundary = prev.is_ascii_lowercase()
                || prev.is_ascii_digit()
                || (prev.is_ascii_uppercase()
                    && chars.get(i + 1).is_some_and(char::is_ascii_lowercase));
            if boundary {
                out.push('_');
            }
        }
        out.push(c);
    }
    let out = out.to_uppercase();
    // Strip a trailing `_CARD` (ports vocab.py commit a082efeef): the
    // registry universe carries it on a subset of entries
    // (`DEBILITATE_CARD`, `BARRICADE_CARD`, …) while scraped C# ids
    // (`CARD.DEBILITATE`) and PascalCase deck names do not; without
    // this those ids canonicalize past each other and fall to OOV.
    match out.strip_suffix("_CARD") {
        Some(base) if !base.is_empty() => base.to_string(),
        _ => out,
    }
}

/// Boss-encounter id normalizer: `"KAISER_CRAB_BOSS"`,
/// `"kaiser_crab_boss"`, `"KaiserCrab"` and `"Kaiser Crab"` all →
/// `"KAISER_CRAB_BOSS"`.
///
/// Ports `vocab.py::canonical_boss`. The plain [`canonical`] is not
/// enough here because the boss identity reaches the encoder from three
/// producers with three shapes: `full_map()`'s boss `id` normally carries
/// the generated `UPPER_SNAKE` encounter id (`THE_KIN_BOSS`), the
/// `EncounterSpec` fallback carries the pool's lower-snake `name`
/// (`kaiser_crab_boss`), and the pinned-boss fallback carries a spaced
/// *display* name (`"Ceremonial Beast"`) with no `_BOSS` suffix at all.
/// Folding every non-alphanumeric run to `_` and appending the suffix
/// when it is missing collapses all three onto the id universe
/// [`sts2_core::run_room_gen::all_boss_encounter_ids`] exports, so a
/// fallback path costs an embedding row rather than silently landing on
/// OOV.
#[must_use]
pub fn canonical_boss(name: &str) -> String {
    let base = canonical(name);
    let mut out = String::with_capacity(base.len() + 5);
    let mut pending_sep = false;
    for c in base.chars() {
        if c.is_ascii_alphanumeric() {
            if pending_sep && !out.is_empty() {
                out.push('_');
            }
            pending_sep = false;
            out.push(c);
        } else {
            pending_sep = true;
        }
    }
    if out.is_empty() || out.ends_with("_BOSS") {
        out
    } else {
        out + "_BOSS"
    }
}

/// name→idx with a reserved OOV slot at `size - 1`.
#[derive(Debug)]
pub struct Registry {
    pub names: Vec<String>,
    idx: BTreeMap<String, usize>,
    pub oov: usize,
    pub size: usize,
    /// The normalizer applied to BOTH the stored names and every query,
    /// so the two sides can never key differently. [`canonical`] for the
    /// card/relic/potion/event universes, [`canonical_boss`] for bosses.
    norm: fn(&str) -> String,
}

impl Registry {
    fn new(names: Vec<String>) -> Self {
        Self::with_norm(names, canonical)
    }

    fn with_norm(names: Vec<String>, norm: fn(&str) -> String) -> Self {
        // Key on the NORMALIZED form of the registry name (not the raw
        // name) so both sides of the `_CARD` normalization agree —
        // `index()` normalizes the query. First writer wins on a
        // (currently nonexistent) collision, keeping indices stable.
        // Mirrors vocab.py `Registry.__init__` (a082efeef).
        let mut idx = BTreeMap::new();
        for (i, n) in names.iter().enumerate() {
            idx.entry(norm(n)).or_insert(i);
        }
        let oov = names.len();
        let size = names.len() + 1;
        Self {
            names,
            idx,
            oov,
            size,
            norm,
        }
    }

    /// Normalize `raw` and look it up, returning [`Self::oov`] on a
    /// miss (mirrors `vocab.py::Registry.index`).
    #[must_use]
    pub fn index(&self, raw: &str) -> usize {
        self.idx.get(&(self.norm)(raw)).copied().unwrap_or(self.oov)
    }
}

/// The six content-derived registry kinds whose embedding-row order is
/// pinned on disk (`src/ppo/vocab_pins/<kind>.txt`).
pub const PINNED_KINDS: [&str; 6] = ["card", "relic_id", "potion", "event", "boss", "enchantment"];

// The pinned row orders. Line N (skipping blanks and `#` comments) is
// embedding row N. See `pinned_order` for why these exist.
const PIN_CARD: &str = include_str!("vocab_pins/card.txt");
const PIN_RELIC_ID: &str = include_str!("vocab_pins/relic_id.txt");
const PIN_POTION: &str = include_str!("vocab_pins/potion.txt");
const PIN_EVENT: &str = include_str!("vocab_pins/event.txt");
const PIN_BOSS: &str = include_str!("vocab_pins/boss.txt");
const PIN_ENCHANTMENT: &str = include_str!("vocab_pins/enchantment.txt");

/// Raw pin-file text for `kind`, or `None` for a kind that has no pin.
#[must_use]
pub fn pinned_order(kind: &str) -> Option<&'static str> {
    match kind {
        "card" => Some(PIN_CARD),
        "relic_id" => Some(PIN_RELIC_ID),
        "potion" => Some(PIN_POTION),
        "event" => Some(PIN_EVENT),
        "boss" => Some(PIN_BOSS),
        "enchantment" => Some(PIN_ENCHANTMENT),
        _ => None,
    }
}

/// Ids of a pin file in row order — non-empty, non-comment lines, trimmed.
pub fn parse_pin(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
}

/// The game build whose id universe the registries span: v0.111.0, whose
/// `all_for` is the union of the v0.107.1 and v0.111.0 ids (each
/// build's `all_for` only appends; ids a later build removed, such as
/// v0.111.0's Scare, stay in the v0.107.1 `all()` it extends).
///
/// GH #799: the registries used to read the v0.107.1 universe (`CardId::all`,
/// `RelicId::all`, `PotionKind::ALL`), which kept every v0.111.0-only id on the
/// OOV row. They now span both builds. The pin files are append-only, so every
/// v0.107.1 id keeps its row and the 0.111 ids sit after them. A v0.107.1 run
/// never produces a 0.111 id, so on that build the only encoding change is the
/// OOV/PAD index moving to the new tail, which a champion re-exported with
/// `_widen_padded_embedding` (PPO `model.py`) reads identically.
///
/// It stops at v0.111.0: the v0.113.0-only cards, relics and events
/// (`V0113_ONLY`) stay on the OOV row. Giving them rows widens every
/// embedding, which needs every champion re-exported (GH #544) — a separate
/// decision from porting the build.
pub const VOCAB_BUILD: GameVersion = GameVersion::V0_111_0;

/// The event universe: every [`EventId`] but the v0.113.0-only ones (see
/// [`VOCAB_BUILD`]).
fn event_universe() -> impl Iterator<Item = &'static EventId> {
    EventId::all()
        .iter()
        .filter(|e| !EventId::V0113_ONLY.contains(e))
}

/// Every potion of every build, in declaration order (v0.107.1 first).
fn potion_universe() -> impl Iterator<Item = &'static PotionKind> {
    PotionKind::ALL.iter().chain(PotionKind::V0111_ONLY.iter())
}

/// The raw, sorted content universe for one of [`PINNED_KINDS`] — what the
/// engine can actually produce today, before the pin imposes an order.
/// [`registry_names`] is this filtered through the pin; the difference
/// between the two is what `vocab_pins_cover_the_content` reports.
///
/// The universe is the **union of every supported game build** (GH #799):
/// one net serves every build, so a v0.111.0-only id needs a row of its own
/// rather than the OOV row. See [`VOCAB_BUILD`].
#[must_use]
pub fn content_ids(kind: &str) -> Option<BTreeSet<&'static str>> {
    Some(match kind {
        "card" => CardId::all_for(VOCAB_BUILD)
            .iter()
            .map(|k| k.to_python_name())
            .collect(),
        "potion" => potion_universe().map(|k| k.python_name()).collect(),
        // v0.111.0 adds no event (the decomp's `Models.Events` file list is
        // identical across the two builds).
        "event" => event_universe().map(|k| k.python_name()).collect(),
        "relic_id" => RelicId::all_for(VOCAB_BUILD)
            .iter()
            .map(|k| k.to_python_name())
            .collect(),
        // Straight from the generator's own `RoomKind::Boss` filter so the
        // vocab cannot drift from the content (see `all_boss_encounter_ids`).
        "boss" => sts2_core::run_room_gen::all_boss_encounter_ids()
            .into_iter()
            .collect(),
        // GH #545: the enchantment stamped by a pending deck-enchant pick.
        // Perfect Fit, Sharp, Adroit and Goopy featurized identically before
        // this — and Goopy is a downside that inverts which card you want.
        "enchantment" => sts2_core::cards::EnchantmentId::ALL
            .iter()
            .map(|e| e.python_name())
            .collect(),
        _ => return None,
    })
}

/// Return the deterministic canonical-name list for one of the six PPO
/// registry kinds (`"card"`, `"relic_id"`, `"potion"`, `"event"`, `"boss"`,
/// `"enchantment"`), or `None` for anything else. The single source for
/// `state_dict.rs::registry_names_for`'s matching arms.
///
/// # Why this is pin-ordered rather than sorted
///
/// The position of a name in this list **is** its embedding row, and the
/// registries are derived from the content on purpose so they cannot drift
/// from what the engine can produce. Sorting made those two properties fight
/// each other: inserting one id renumbered every id after it, so a deployed
/// champion kept reading row *k* while row *k* now meant a different boss.
/// Porting Underdocks as the second Act 1 moved five of the nine boss rows
/// that way (GH #544).
///
/// So the order comes from an append-only pin file instead. Pinned ids hold
/// their rows forever — including ids the content has since dropped, whose
/// rows go unreachable rather than shifting everything behind them. Content
/// ids the pin does not list are appended in sorted order, and
/// `vocab_pins_cover_the_content` fails with the exact lines to add so the
/// pin can never quietly fall behind.
#[must_use]
pub fn registry_names(kind: &str) -> Option<Vec<String>> {
    let content = content_ids(kind)?;
    let pin = pinned_order(kind)?;
    let mut out: Vec<String> = Vec::with_capacity(content.len());
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for id in parse_pin(pin) {
        if seen.insert(id) {
            out.push(id.to_string());
        }
    }
    for id in &content {
        if !seen.contains(id) {
            out.push((*id).to_string());
        }
    }
    Some(out)
}

/// Return `(debug_form, python_name)` for every id of one of the four
/// PPO registry kinds (`"card"`, `"relic_id"`, `"potion"`, `"event"`),
/// or `None` for anything else. The Debug form is what `{:?}` dumps
/// emit (`master_deck_detailed`, inventory potion ids, …); the python
/// name is what `list_registry_names` / reward-option ids carry. One
/// entry per enum variant, unsorted and undeduped — unlike
/// [`registry_names`] this preserves the raw per-id universe so a
/// round-trip test can probe every emitted form (GH#286).
#[must_use]
pub fn registry_id_forms(kind: &str) -> Option<Vec<(String, String)>> {
    let forms: Vec<(String, String)> = match kind {
        // The same every-build universe as `content_ids` (GH #799).
        "card" => CardId::all_for(VOCAB_BUILD)
            .iter()
            .map(|k| (format!("{k:?}"), k.to_python_name().to_string()))
            .collect(),
        "potion" => potion_universe()
            .map(|k| (format!("{k:?}"), k.python_name().to_string()))
            .collect(),
        "event" => event_universe()
            .map(|k| (format!("{k:?}"), k.python_name().to_string()))
            .collect(),
        "relic_id" => RelicId::all_for(VOCAB_BUILD)
            .iter()
            .map(|k| (format!("{k:?}"), k.to_python_name().to_string()))
            .collect(),
        _ => return None,
    };
    Some(forms)
}

fn build_registry(kind: &str) -> Registry {
    Registry::new(
        registry_names(kind).unwrap_or_else(|| panic!("unknown PPO registry kind `{kind}`")),
    )
}

/// The card-id registry (`CardId::all()` canonical names, ~579 entries).
#[must_use]
pub fn cards() -> &'static Registry {
    static CARDS: OnceLock<Registry> = OnceLock::new();
    CARDS.get_or_init(|| build_registry("card"))
}

/// The full run-side relic registry (`RelicId::all()`, ~290 entries) —
/// NOT the ~111-entry combat `RelicKind` subset (kind `"relic"`).
#[must_use]
pub fn relics() -> &'static Registry {
    static RELICS: OnceLock<Registry> = OnceLock::new();
    RELICS.get_or_init(|| build_registry("relic_id"))
}

/// The potion-kind registry (`PotionKind::ALL`, 63 entries).
#[must_use]
pub fn potions() -> &'static Registry {
    static POTIONS: OnceLock<Registry> = OnceLock::new();
    POTIONS.get_or_init(|| build_registry("potion"))
}

/// The run-event registry (`EventId::all()` python names, minus the
/// v0.113.0-only events).
#[must_use]
pub fn events() -> &'static Registry {
    static EVENTS: OnceLock<Registry> = OnceLock::new();
    EVENTS.get_or_init(|| build_registry("event"))
}

/// The boss-encounter registry (9 entries) — the identity of the boss
/// waiting at the top of the current act's map.
///
/// This is the only registry keyed by [`canonical_boss`] rather than
/// [`canonical`]; see that function for why the boss identity needs a
/// looser normalizer than the rest.
#[must_use]
pub fn bosses() -> &'static Registry {
    static BOSSES: OnceLock<Registry> = OnceLock::new();
    BOSSES.get_or_init(|| {
        Registry::with_norm(
            registry_names("boss").expect("`boss` is a known PPO registry kind"),
            canonical_boss,
        )
    })
}

/// The enchantment registry (GH #545) — which enchantment a pending
/// deck-enchant pick will stamp on the card you choose.
#[must_use]
pub fn enchantments() -> &'static Registry {
    static ENCHANTMENTS: OnceLock<Registry> = OnceLock::new();
    ENCHANTMENTS.get_or_init(|| build_registry("enchantment"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_normalizes_both_forms() {
        assert_eq!(canonical("CARD.ULTIMATE_STRIKE"), "ULTIMATE_STRIKE");
        assert_eq!(canonical("UltimateStrike"), "ULTIMATE_STRIKE");
        assert_eq!(canonical("HTTPServer2Go"), "HTTP_SERVER2_GO"); // camel-boundary edge parity
    }

    /// Every shape the boss identity can reach the encoder in must land
    /// on the SAME embedding row: the generated `UPPER_SNAKE` id (the
    /// normal path), the `EncounterSpec` lower-snake pool name, the
    /// PascalCase form, and the spaced display name the pinned-boss
    /// fallback emits.
    #[test]
    fn canonical_boss_folds_every_producer_shape_onto_one_id() {
        for raw in [
            "KAISER_CRAB_BOSS",
            "kaiser_crab_boss",
            "KaiserCrab",
            "Kaiser Crab",
            "Kaiser Crab Boss",
        ] {
            assert_eq!(canonical_boss(raw), "KAISER_CRAB_BOSS", "{raw}");
        }
        assert_eq!(canonical_boss(""), "");
    }

    #[test]
    fn boss_registry_covers_every_rollable_boss_and_oovs_the_rest() {
        let b = bosses();
        assert_eq!(b.size, b.names.len() + 1);
        // 12 since the Underdocks port (GH #544): 9 pre-existing + Lagavulin
        // Matriarch, Soul Fysh, Waterfall Giant. The three APPEND — the pin
        // holds rows 0..8 fixed — so a champion re-baked at the new width
        // reads the same row for every boss it was trained on.
        assert_eq!(b.names.len(), 12, "{:?}", b.names);
        // Everything map generation can roll resolves, in every shape.
        for id in sts2_core::run_room_gen::all_boss_encounter_ids() {
            assert!(b.index(id) < b.oov, "{id} is OOV");
            assert_eq!(
                b.index(&id.to_lowercase()),
                b.index(id),
                "{id} case-folds to a different row"
            );
        }
        assert_eq!(b.index("NOT_A_BOSS_XYZ_BOSS"), b.oov);
        // An empty/absent boss id must NOT collide with a real boss row.
        assert_eq!(b.index(""), b.oov);
    }

    /// The pin files must list every id the content can produce. When they
    /// do not, this is the ONLY place that says so — an unpinned id still
    /// gets an index (appended, sorted), but the NEXT addition would shift
    /// it, which is exactly the silent renumbering the pins exist to stop.
    ///
    /// Fix by running `cargo run -p sts2-runai --example dump_vocab_pins`
    /// and committing the appended lines. Never reorder or delete a line.
    #[test]
    fn vocab_pins_cover_the_content() {
        for kind in PINNED_KINDS {
            let pinned: BTreeSet<&str> = parse_pin(pinned_order(kind).unwrap()).collect();
            let missing: Vec<&str> = content_ids(kind)
                .unwrap()
                .into_iter()
                .filter(|id| !pinned.contains(id))
                .collect();
            assert!(
                missing.is_empty(),
                "`{kind}` content is ahead of its pin file. Append these lines to \
                 src/ppo/vocab_pins/{kind}.txt (in this order) via \
                 `cargo run -p sts2-runai --example dump_vocab_pins`:\n{}",
                missing.join("\n")
            );
        }
    }

    /// Row counts of the v0.107.1-only vocab every champion before GH #799 was
    /// exported at. Those rows must keep their ids forever.
    const V0107_ROWS: [(&str, usize); 3] = [("card", 580), ("relic_id", 297), ("potion", 63)];

    /// GH #799: every v0.111.0-only id has a real (non-OOV) row, and those rows
    /// sit AFTER every v0.107.1 row, so the union re-lays nothing a pre-0.111
    /// champion learned.
    #[test]
    fn registries_span_the_v0111_ids_after_the_v0107_rows_v0111() {
        let v0111: [(&str, &Registry, Vec<&str>); 3] = [
            (
                "card",
                cards(),
                CardId::V0111_ONLY
                    .iter()
                    .map(|k| k.to_python_name())
                    .collect(),
            ),
            (
                "relic_id",
                relics(),
                RelicId::V0111_ONLY
                    .iter()
                    .map(|k| k.to_python_name())
                    .collect(),
            ),
            (
                "potion",
                potions(),
                PotionKind::V0111_ONLY
                    .iter()
                    .map(|k| k.python_name())
                    .collect(),
            ),
        ];
        for ((kind, reg, ids), (kind_rows, v0107_rows)) in v0111.iter().zip(V0107_ROWS) {
            assert_eq!(*kind, kind_rows);
            assert_eq!(reg.names.len(), v0107_rows + ids.len(), "{kind}");
            for id in ids {
                let row = reg.index(id);
                assert!(row != reg.oov, "{kind}: {id} is OOV");
                assert!(row >= v0107_rows, "{kind}: {id} took v0.107.1 row {row}");
            }
        }
    }

    /// The v0.107.1 universe still maps onto the first rows, id for id: the
    /// sorted v0.107.1 content is exactly the first `V0107_ROWS` pin lines.
    #[test]
    fn v0107_ids_keep_their_rows() {
        let v0107: [(&str, BTreeSet<&str>); 3] = [
            (
                "card",
                CardId::all().iter().map(|k| k.to_python_name()).collect(),
            ),
            (
                "relic_id",
                RelicId::all().iter().map(|k| k.to_python_name()).collect(),
            ),
            (
                "potion",
                PotionKind::ALL.iter().map(|k| k.python_name()).collect(),
            ),
        ];
        for ((kind, ids), (_, rows)) in v0107.iter().zip(V0107_ROWS) {
            let names = registry_names(kind).unwrap();
            let head: BTreeSet<&str> = names[..rows].iter().map(String::as_str).collect();
            // Every v0.107.1 id is in the head (the head may also hold ids the
            // content has since dropped; those rows stay put, unreachable).
            assert!(ids.is_subset(&head), "{kind}: a v0.107.1 id left its row");
        }
    }

    /// A duplicated pin line would silently consume a row and shift every id
    /// behind it — the same damage as a reorder, but harder to spot in review.
    #[test]
    fn vocab_pins_have_no_duplicate_rows() {
        for kind in PINNED_KINDS {
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            for id in parse_pin(pinned_order(kind).unwrap()) {
                assert!(seen.insert(id), "`{kind}` pin lists {id} twice");
            }
        }
    }

    /// `registry_names` must reproduce the pin order verbatim for the ids the
    /// content still has, and only append after it. Asserting the ORDER (not
    /// just the membership `vocab_pins_cover_the_content` checks) is what
    /// makes the pin load-bearing: a refactor that went back to sorting would
    /// keep every membership test green.
    #[test]
    fn registry_names_follows_the_pin_order() {
        for kind in PINNED_KINDS {
            let names = registry_names(kind).unwrap();
            let pinned: Vec<&str> = parse_pin(pinned_order(kind).unwrap()).collect();
            assert_eq!(
                names[..pinned.len()],
                pinned.iter().map(|s| (*s).to_string()).collect::<Vec<_>>()[..],
                "`{kind}` embedding rows no longer match the pin"
            );
        }
    }

    /// The nine boss rows every champion deployed before the Underdocks port
    /// was trained on, in the order it was trained on them. This is the
    /// compatibility anchor: `emb_boss` row 4 means `QUEEN_BOSS` to those nets
    /// forever, so these nine lines must open the boss pin and never move.
    /// Underdocks' three bosses append after them (GH #544).
    #[test]
    fn boss_pin_opens_with_the_nine_pre_underdocks_rows() {
        let pinned: Vec<&str> = parse_pin(PIN_BOSS).collect();
        assert_eq!(
            pinned[..9],
            [
                "AEONGLASS_BOSS",
                "CEREMONIAL_BEAST_BOSS",
                "KAISER_CRAB_BOSS",
                "KNOWLEDGE_DEMON_BOSS",
                "QUEEN_BOSS",
                "TEST_SUBJECT_BOSS",
                "THE_INSATIABLE_BOSS",
                "THE_KIN_BOSS",
                "VANTOM_BOSS",
            ]
        );
    }

    #[test]
    fn registries_have_oov_and_match_python_kinds() {
        let c = cards();
        assert_eq!(c.size, c.names.len() + 1);
        assert_eq!(c.index("NOT_A_CARD_XYZ"), c.oov);
        assert!(c.index("BASH") < c.oov);
        assert!(relics().names.len() > 200); // relic_id (~290), not the ~111 combat subset
        assert!(events().names.len() > 40);
    }

    /// GH#286: every id the engine can emit — in EVERY form it emits it
    /// (Debug/PascalCase from `master_deck_detailed`-style `{:?}` dumps,
    /// raw `python_name`, the `PREFIX.python_name` reward form, and for
    /// cards the scraped C# form whose ids carry no `_CARD` suffix) —
    /// must resolve to a non-OOV embedding index.
    #[test]
    fn every_id_round_trips_non_oov_in_every_emitted_form() {
        for (kind, reg) in [
            ("card", cards()),
            ("relic_id", relics()),
            ("potion", potions()),
            ("event", events()),
        ] {
            let forms = registry_id_forms(kind).expect("known PPO registry kind");
            assert!(!forms.is_empty(), "{kind}: empty id-forms universe");
            let prefix = match kind {
                "card" => "CARD",
                "relic_id" => "RELIC",
                "potion" => "POTION",
                "event" => "EVENT",
                _ => unreachable!(),
            };
            let mut misses: Vec<String> = Vec::new();
            for (debug, py) in &forms {
                let mut queries = vec![debug.clone(), py.clone(), format!("{prefix}.{py}")];
                if kind == "card" {
                    // Scraped .run files use the C# id, which namespaces by
                    // `CARD.` and never carries the Rust-only `_CARD`
                    // disambiguation suffix (see pycli `_csharp_card_id`).
                    let cs = py
                        .strip_suffix("_CARD")
                        .filter(|s| !s.is_empty())
                        .unwrap_or(py);
                    queries.push(format!("CARD.{cs}"));
                }
                for q in queries {
                    if reg.index(&q) == reg.oov {
                        misses.push(q);
                    }
                }
            }
            assert!(
                misses.is_empty(),
                "{kind}: {} emitted forms landed in the OOV slot, e.g. {:?}",
                misses.len(),
                &misses[..misses.len().min(20)]
            );
        }
    }

    /// First-writer-wins on canonical collision keeps indices stable but
    /// silently merges two ids into one embedding row — assert the
    /// current universes have none, so a future name addition that
    /// collides fails loud here instead.
    #[test]
    fn no_canonical_collisions_within_any_registry() {
        for (kind, reg) in [
            ("card", cards()),
            ("relic_id", relics()),
            ("potion", potions()),
            ("event", events()),
        ] {
            let canon: BTreeSet<String> = reg.names.iter().map(|n| canonical(n)).collect();
            assert_eq!(
                canon.len(),
                reg.names.len(),
                "{kind}: canonical() collision within the registry universe"
            );
        }
    }

    /// The card universe must span all five characters (starter Strikes
    /// as witnesses) — guards against a registry source that silently
    /// drops a character's card pool.
    #[test]
    fn card_universe_spans_all_five_characters() {
        let c = cards();
        for s in [
            "STRIKE_IRONCLAD",
            "STRIKE_SILENT",
            "STRIKE_DEFECT",
            "STRIKE_REGENT",
            "STRIKE_NECROBINDER",
        ] {
            assert!(c.index(s) < c.oov, "{s} is OOV");
        }
    }

    #[test]
    fn fixed_and_surface_orders_match_python() {
        assert_eq!(FIXED_KINDS[5], "shop_remove");
        assert_eq!(fixed_index("nonexistent"), 0); // falls back to "none"
        assert_eq!(SURFACES[0], "path");
        assert_eq!(surface_index("weird"), 14); // "unknown"
        assert_eq!(room_index("rest_site"), 6);
        assert_eq!(room_index("bogus"), 9); // ROOM_OOV
    }

    /// GH #545: the enchantment identity needs an embedding row, and its
    /// order must be an append-only pin for the same reason the boss vocab
    /// is one — the position IS the row a deployed champion learned.
    #[test]
    fn enchantment_registry_is_pin_ordered_and_covers_the_content() {
        let names = registry_names("enchantment").expect("enchantment is a PPO registry kind");
        assert_eq!(names.first().map(String::as_str), Some("ROYALLY_APPROVED"));
        let content = content_ids("enchantment").expect("enchantment has content ids");
        for id in &content {
            assert!(names.iter().any(|n| n == id), "{id} missing from the pin");
        }
        let mut sorted = names.clone();
        sorted.sort();
        assert_ne!(
            names, sorted,
            "the enchantment pin is declaration-ordered, not sorted; if this ever \
             matches, the pin was rewritten and every learned row moved"
        );
        assert_eq!(
            enchantments().index("GOOPY"),
            names.iter().position(|n| n == "GOOPY").unwrap()
        );
    }
}
