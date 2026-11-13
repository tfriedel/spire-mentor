//! Dual-form identifier name lookup — bead `sts2-rs-e5j9` (P9 / cliext C1).
//!
//! The CLI extension plan (`docs/plans/2026-05-14-extend-sts2-cli-...md`
//! §3.2/§3.3/§3.4 + §5 Wave C1) needs to accept BOTH the canonical
//! Rust `PascalCase` variant name (e.g. `BurningBlood`) AND the human
//! display string (e.g. `"Burning Blood"`, case-insensitive) on the
//! CLI. This module centralizes the dual-form lookup so the three
//! identifier enums share one implementation.
//!
//! ## Why a single helper
//!
//! Each of [`sts2_core::cards::CardId`], [`sts2_core::relics::RelicId`],
//! and [`sts2_core::PotionKind`] already exposes:
//!
//! - an iterator over every variant (`CardId::all()`, `RelicId::all()`,
//!   `PotionKind::ALL`), AND
//! - an `UPPER_SNAKE_CASE` Python parity name
//!   (`to_python_name(self)` / `python_name(self)`).
//!
//! The Rust `PascalCase` variant name (e.g. `BurningBlood`), the Python
//! parity name (`BURNING_BLOOD`), and the display string (`Burning
//! Blood`) all collapse to the same canonical form under
//! `lowercase + strip-non-alphanumeric`. So a single normalized
//! comparison against the Python-parity table covers all four forms
//! the bead requires:
//!
//! - `BurningBlood` (`PascalCase` variant)
//! - `Burning Blood` (display string)
//! - `burning blood` (case-insensitive display)
//! - `burning_blood` (snake-case)
//!
//! Output side (`suggest`) reconstructs the canonical `PascalCase` form
//! from the Python parity name, since that is the CLI's
//! recommended-form per plan §3.3.
//!
//! ## Orphan rule
//!
//! `CardId` / `RelicId` / `PotionKind` live in `sts2-core`; this
//! module lives in `sts2-content`. Rust's orphan rules forbid
//! `impl FromStr` for foreign types from a downstream crate, so the
//! bead's "Notes" fallback path is taken: free functions
//! ([`card_id_from_str`], [`relic_id_from_str`],
//! [`potion_kind_from_str`]) instead of `FromStr` impls.

use sts2_core::cards::CardId;
use sts2_core::relics::RelicId;
use sts2_core::PotionKind;

/// Maximum number of did-you-mean suggestions returned by [`suggest`].
const MAX_SUGGESTIONS: usize = 3;

/// Levenshtein-distance cap for inclusion in [`suggest`] output. Larger
/// values produce noisier suggestions; smaller values miss obvious
/// typos. 5 is a good middle ground for 8–20-character identifiers.
const SUGGEST_MAX_DISTANCE: usize = 5;

/// Lowercase + strip non-alphanumeric.
///
/// This is the canonical form used for dual-form lookup. The four
/// shapes `BurningBlood`, `Burning Blood`, `burning blood`,
/// `burning_blood` all normalize to `burningblood`.
fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        }
    }
    out
}

/// Convert an `UPPER_SNAKE_CASE` Python parity name to `PascalCase`.
///
/// `BURNING_BLOOD` → `BurningBlood`. Used both for restoring the
/// canonical CLI form in [`suggest`] output and for the
/// `PascalCase`-as-display reconstruction the plan calls out (§3.3
/// "`PascalCase` is the canonical CLI shape").
fn snake_to_pascal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut next_upper = true;
    for ch in s.chars() {
        if ch == '_' {
            next_upper = true;
        } else if next_upper {
            out.extend(ch.to_uppercase());
            next_upper = false;
        } else {
            out.extend(ch.to_lowercase());
        }
    }
    out
}

/// Insert a space before every uppercase letter that follows a
/// lowercase letter (`PascalCase` → `Pascal Case`). Produces the
/// "display string" form the bead's acceptance criteria reference.
fn pascal_to_display(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    let mut prev_lower = false;
    for ch in s.chars() {
        if ch.is_ascii_uppercase() && prev_lower {
            out.push(' ');
        }
        out.push(ch);
        prev_lower = ch.is_ascii_lowercase() || ch.is_ascii_digit();
    }
    out
}

/// Dual-form lookup: resolve `input` to one of `values` by matching
/// against the matching `python_names` (`UPPER_SNAKE_CASE` Python parity
/// names) under the [`normalize`] canonical form.
///
/// Accepts both the Rust `PascalCase` variant name AND the display
/// string. See module docs for why we don't need a separate display
/// table.
///
/// # Panics
///
/// Panics if `values.len() != python_names.len()`. Always called with
/// parallel arrays produced by [`all_card_id_names`] etc., so this is
/// a programmer error if it ever fires.
pub fn name_lookup<T: Copy>(input: &str, values: &[T], python_names: &[&str]) -> Option<T> {
    assert_eq!(
        values.len(),
        python_names.len(),
        "name_lookup: values and python_names must be parallel"
    );
    let normalized_input = normalize(input);
    if normalized_input.is_empty() {
        return None;
    }
    for (value, python_name) in values.iter().zip(python_names.iter()) {
        if normalize(python_name) == normalized_input {
            return Some(*value);
        }
    }
    None
}

/// Levenshtein distance, hand-rolled to avoid pulling `strsim` in for
/// one bead. Uses two rolling rows — `O(min(a, b))` space,
/// `O(a * b)` time. Adequate for ≤ ~600 variants × handful of
/// suggestions per CLI invocation.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let (a_len, b_len) = (a_bytes.len(), b_bytes.len());
    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }
    let mut prev: Vec<usize> = (0..=b_len).collect();
    let mut curr: Vec<usize> = vec![0; b_len + 1];
    for i in 1..=a_len {
        curr[0] = i;
        for j in 1..=b_len {
            let cost = usize::from(a_bytes[i - 1] != b_bytes[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b_len]
}

/// Top-3 did-you-mean suggestions for `input` over `python_names`,
/// returned in canonical `PascalCase` form (the recommended CLI shape).
///
/// Ties broken by the order entries appear in `python_names` (which
/// follows the Python source-of-truth declaration order).
pub fn suggest(input: &str, python_names: &[&str]) -> Vec<String> {
    let normalized_input = normalize(input);
    let mut scored: Vec<(usize, &&str)> = python_names
        .iter()
        .map(|name| (levenshtein(&normalized_input, &normalize(name)), name))
        .filter(|(d, _)| *d <= SUGGEST_MAX_DISTANCE)
        .collect();
    scored.sort_by_key(|(d, _)| *d);
    scored
        .into_iter()
        .take(MAX_SUGGESTIONS)
        .map(|(_, name)| snake_to_pascal(name))
        .collect()
}

// -------------------------------------------------------------------
// Per-enum tables + free-function front doors
// -------------------------------------------------------------------

/// Build a `(values, python_names)` pair for [`CardId`].
fn card_id_tables() -> (Vec<CardId>, Vec<&'static str>) {
    let values: Vec<CardId> = CardId::all().to_vec();
    let names: Vec<&'static str> = values.iter().map(|v| v.to_python_name()).collect();
    (values, names)
}

/// Resolve a CLI/scenario string to a [`CardId`], accepting both the
/// `PascalCase` variant form (e.g. `BurningBlood`) and the human display
/// form (e.g. `"Burning Blood"`, case-insensitive).
///
/// Also accepts the sts2log-emitted suffix-stripped form (e.g.
/// `RUPTURE` for `CardId::RuptureCard`). The Python pin spells these
/// cards as `RUPTURE_CARD` / `VICIOUS_CARD` / etc., but real-run log
/// dumps emit the un-suffixed token because game UI strips it. Bead
/// `sts2-rs-u5zc` lets the resolver fall back on stripping the
/// trailing `_CARD` from each python name before comparing, so log
/// fixtures load without the scraper having to round-trip the
/// suffix.
#[must_use]
pub fn card_id_from_str(input: &str) -> Option<CardId> {
    let (values, names) = card_id_tables();
    if let Some(hit) = name_lookup(input, &values, &names) {
        return Some(hit);
    }
    // Suffix-stripped fallback: compare `normalize(name without trailing _CARD)`
    // against `normalize(input)`. Card python_names that don't carry the
    // suffix are still matched by the first pass above.
    let normalized_input = normalize(input);
    if normalized_input.is_empty() {
        return None;
    }
    for (value, python_name) in values.iter().zip(names.iter()) {
        if let Some(stripped) = python_name.strip_suffix("_CARD") {
            if normalize(stripped) == normalized_input {
                return Some(*value);
            }
        }
    }
    // GH #793: the v0.111.0-only ids are outside `CardId::all` (the v0.107.1
    // universe the tables above walk). A v0.111.0 scenario deck names them, so
    // resolve them last — after every v0.107.1 name, which they cannot shadow.
    // The v0.113.0-only ids follow the same rule.
    let v0111: Vec<CardId> = CardId::V0111_ONLY
        .iter()
        .chain(CardId::V0113_ONLY.iter())
        .copied()
        .collect();
    let v0111_names: Vec<&'static str> = v0111.iter().map(|v| v.to_python_name()).collect();
    if let Some(hit) = name_lookup(input, &v0111, &v0111_names) {
        return Some(hit);
    }
    for (value, python_name) in v0111.iter().zip(v0111_names.iter()) {
        if let Some(stripped) = python_name.strip_suffix("_CARD") {
            if normalize(stripped) == normalized_input {
                return Some(*value);
            }
        }
    }
    None
}

/// Top-3 did-you-mean suggestions for an unresolved [`CardId`] input.
#[must_use]
pub fn suggest_card_id(input: &str) -> Vec<String> {
    let (_, names) = card_id_tables();
    suggest(input, &names)
}

// ─── Player-class-hinted card resolution (cliext C2 / sts2-rs-63bi) ───
//
// CLI deck strings use bare `Strike` / `Defend` tokens; map them to the
// class-specific CardId variant via a hint. Returns the full
// [`CardSpec`] (POD metadata) for callers that need cost / type / damage
// rather than just the [`CardId`].
//
// The CardId → CardSpec lookup is backed by a small hand-maintained
// factory table — only deck-relevant cards need to be materializable
// today. Extend the registry when new cards become eligible for the
// `--deck` flag.

/// Hint for bare `Strike` / `Defend` token resolution in
/// [`resolve_card_name`], and class filter for
/// [`deck_card_spec_pool_for_class`] (bead `sts2-rs-9yoh` / cliext F4).
/// Defect added in cliext H4 (bead `sts2-rs-pjln`) once the generated
/// Defect factories were registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerClassHint {
    Ironclad,
    Silent,
    Defect,
}

impl PlayerClassHint {
    /// Every class with deck-eligible cards in
    /// [`deck_card_spec_registry`]. Extend as P6-5/P6-7 land the
    /// Necrobinder / Regent rows.
    pub const SUPPORTED_FOR_RANDOM_DECK: &'static [PlayerClassHint] = &[
        PlayerClassHint::Ironclad,
        PlayerClassHint::Silent,
        PlayerClassHint::Defect,
    ];

    /// Canonical lowercase tag used by CLI flag parsing.
    #[must_use]
    pub fn as_tag(self) -> &'static str {
        match self {
            PlayerClassHint::Ironclad => "ironclad",
            PlayerClassHint::Silent => "silent",
            PlayerClassHint::Defect => "defect",
        }
    }

    /// Parse a case-insensitive tag (`ironclad` / `silent` / `defect`).
    /// Returns `None` on unknown input — the caller surfaces a
    /// user-facing error with the supported list.
    #[must_use]
    pub fn from_tag(s: &str) -> Option<PlayerClassHint> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ironclad" => Some(PlayerClassHint::Ironclad),
            "silent" => Some(PlayerClassHint::Silent),
            "defect" => Some(PlayerClassHint::Defect),
            _ => None,
        }
    }
}

/// Error returned by [`resolve_card_name`] when the input does not map
/// to any known card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownCard {
    pub input: String,
    /// Top-3 did-you-mean suggestions (canonical `PascalCase` form).
    pub suggestions: Vec<String>,
}

/// `CardSpec` factory table for deck-eligible cards. Hand-maintained;
/// keyed by [`CardId`]. Extend when a new card becomes useable via
/// `--deck`.
///
/// Coverage target (bead `sts2-rs-xzeu` / cliext F2): every Ironclad
/// and Silent common card that has a POD spec in the generated card
/// modules at pin `01c094d`. Cards listed in the F2 spec that are not
/// yet ported (e.g. `Flex`, `Cleave`, `Clothesline`, `WarCry`,
/// `WildStrike`, `Bane`, `FlyingKnee`, `QuickSlash`, `SlicingStrike`,
/// `SneakyStrike`, `Catalyst`, `Outmaneuver` as a silent common) are
/// intentionally omitted — they have no generated factory.
/// Curated registry of starter cards plus the demo presets. These take
/// precedence over generated entries on `CardId` collision — e.g. the
/// hand-rolled `ironclad_basic::strike` is preferred over the
/// generated `ic::strike_ironclad` because the hand-rolled version
/// carries Phase-4 listener bindings the generated POD spec doesn't.
///
/// Cliext H4 (bead `sts2-rs-pjln`) merges this list with the
/// 354-entry `spec_index` slices to expose every dispatch-supported
/// card via `--deck` and `list cards`.
fn deck_card_spec_registry() -> Vec<(
    CardId,
    Option<PlayerClassHint>,
    fn() -> crate::cards::CardSpec,
)> {
    use crate::cards::ironclad_basic;
    use crate::cards::spec_index::{
        COLORLESS_SPECS, DEFECT_SPECS, IRONCLAD_SPECS, NECROBINDER_SPECS, REGENT_SPECS,
        SILENT_SPECS, STATUS_SPECS,
    };
    use PlayerClassHint::{Ironclad, Silent};

    // Curated entries first; the merge step below skips any generated
    // factory whose CardId is already present here.
    let curated: Vec<(
        CardId,
        Option<PlayerClassHint>,
        fn() -> crate::cards::CardSpec,
    )> = vec![
        // Ironclad starters (hand-rolled factories).
        (
            CardId::StrikeIronclad,
            Some(Ironclad),
            ironclad_basic::strike,
        ),
        (
            CardId::DefendIronclad,
            Some(Ironclad),
            ironclad_basic::defend,
        ),
        (CardId::Bash, Some(Ironclad), ironclad_basic::bash),
        // AscendersBane is a class-neutral Curse card (CardType::Curse,
        // CardRarity::Curse) — its factory lives in ironclad_basic for
        // historical reasons but the card is colorless (available to all
        // classes). Tag None so it does not leak into Ironclad's
        // class-locked random-deck pool (root cause of sts2-rs-ttte).
        (CardId::AscendersBane, None, ironclad_basic::ascenders_bane),
        // Silent starters (generated POD specs).
        (
            CardId::StrikeSilent,
            Some(Silent),
            crate::cards::silent::strike_silent::spec,
        ),
        (
            CardId::DefendSilent,
            Some(Silent),
            crate::cards::silent::defend_silent::spec,
        ),
        (
            CardId::Neutralize,
            Some(Silent),
            crate::cards::silent::neutralize::spec,
        ),
        (
            CardId::Survivor,
            Some(Silent),
            crate::cards::silent::survivor::spec,
        ),
        // Common deck-demo Ironclad cards.
        (
            CardId::Anger,
            Some(Ironclad),
            crate::cards::ironclad::anger::spec,
        ),
        (
            CardId::Armaments,
            Some(Ironclad),
            crate::cards::ironclad::armaments::spec,
        ),
        (
            CardId::BodySlam,
            Some(Ironclad),
            crate::cards::ironclad::body_slam::spec,
        ),
        (
            CardId::Headbutt,
            Some(Ironclad),
            crate::cards::ironclad::headbutt::spec,
        ),
        (
            CardId::IronWave,
            Some(Ironclad),
            crate::cards::ironclad::iron_wave::spec,
        ),
        (
            CardId::PerfectedStrike,
            Some(Ironclad),
            crate::cards::ironclad::perfected_strike::spec,
        ),
        (
            CardId::PommelStrike,
            Some(Ironclad),
            crate::cards::ironclad::pommel_strike::spec,
        ),
        (
            CardId::ShrugItOff,
            Some(Ironclad),
            crate::cards::ironclad::shrug_it_off::spec,
        ),
        (
            CardId::SwordBoomerang,
            Some(Ironclad),
            crate::cards::ironclad::sword_boomerang::spec,
        ),
        (
            CardId::Thunderclap,
            Some(Ironclad),
            crate::cards::ironclad::thunderclap::spec,
        ),
        (
            CardId::TrueGrit,
            Some(Ironclad),
            crate::cards::ironclad::true_grit::spec,
        ),
        (
            CardId::TwinStrike,
            Some(Ironclad),
            crate::cards::ironclad::twin_strike::spec,
        ),
        // Demo-only Ironclad uncommon (referenced by
        // `ironclad-strength-demo` preset).
        (
            CardId::Inflame,
            Some(Ironclad),
            crate::cards::ironclad::inflame::spec,
        ),
        // Common deck-demo Silent cards.
        (
            CardId::BladeDance,
            Some(Silent),
            crate::cards::silent::blade_dance::spec,
        ),
        (
            CardId::CloakAndDagger,
            Some(Silent),
            crate::cards::silent::cloak_and_dagger::spec,
        ),
        (
            CardId::DaggerSpray,
            Some(Silent),
            crate::cards::silent::dagger_spray::spec,
        ),
        (
            CardId::DaggerThrow,
            Some(Silent),
            crate::cards::silent::dagger_throw::spec,
        ),
        (
            CardId::DeadlyPoison,
            Some(Silent),
            crate::cards::silent::deadly_poison::spec,
        ),
        (
            CardId::Deflect,
            Some(Silent),
            crate::cards::silent::deflect::spec,
        ),
        (
            CardId::DodgeAndRoll,
            Some(Silent),
            crate::cards::silent::dodge_and_roll::spec,
        ),
        (
            CardId::PiercingWail,
            Some(Silent),
            crate::cards::silent::piercing_wail::spec,
        ),
        (
            CardId::PoisonedStab,
            Some(Silent),
            crate::cards::silent::poisoned_stab::spec,
        ),
        (
            CardId::SuckerPunch,
            Some(Silent),
            crate::cards::silent::sucker_punch::spec,
        ),
        // Bead `sts2-rs-d894` (qt0b followup): Convergence is the
        // only Regent card the current sts2log corpus needs to
        // deck-load. Regent has no `.rs`, so
        // the spec is hand-rolled in `cards/regent.rs::convergence`
        // (colorless_handrolled pattern). Class tag is `None` to
        // keep the card out of class-locked random-deck pools until
        // PlayerClassHint grows a `Regent` variant.
        (CardId::Convergence, None, crate::cards::regent::convergence),
        // Necrobinder Strike/Defend factories (sts2-rs-yein) removed with the
        // card pool for a from-scratch C#-first re-port. Re-add the tuples when
        // the necrobinder card module returns.
        // Bead `sts2-rs-3kyp`: Regent Strike/Defend factories so
        // sts2log Regent fixtures can deck-load. Mirror of yein for
        // the Regent class. Class tag `None` until `PlayerClassHint`
        // grows a `Regent` variant.
        (
            CardId::StrikeRegent,
            None,
            crate::cards::regent::strike_regent,
        ),
        (
            CardId::DefendRegent,
            None,
            crate::cards::regent::defend_regent,
        ),
        (
            CardId::FallingStar,
            None,
            crate::cards::regent::falling_star,
        ),
        (CardId::Venerate, None, crate::cards::regent::venerate),
    ];

    // Build a set of CardIds already covered by curated entries so the
    // merge step doesn't duplicate factories (which would surface as
    // double-counted entries in `deck_card_spec_pool` and confuse
    // class-locked random-deck sampling).
    let covered: std::collections::HashSet<CardId> = curated.iter().map(|(id, _, _)| *id).collect();

    let mut merged = curated;
    // Append generated factories. Class tag is `Some(<class>)` for the
    // three real character classes; `None` for Colorless/Status (those
    // are not class-locked and won't appear in --random-deck-character).
    for (class, slice) in [
        (Some(Ironclad), IRONCLAD_SPECS),
        (Some(Silent), SILENT_SPECS),
        (Some(PlayerClassHint::Defect), DEFECT_SPECS),
        // Regent has all 88 cards in REGENT_SPECS but no `PlayerClassHint`
        // variant yet, so tag `None` (mirrors the curated Regent starters
        // above): the cards become resolvable by `resolve_card_name` — which
        // the Regent parity coverage pack needs to deck-load each card under
        // test — without leaking into class-locked --random-deck pools.
        (None, REGENT_SPECS),
        // Necrobinder, same as Regent: all 86 cards in NECROBINDER_SPECS but
        // no `PlayerClassHint` variant yet. Without this row every Necrobinder
        // card was unresolvable by `resolve_card_name` (the parity coverage /
        // v2 sweeps for Necrobinder bucketed 100% HARNESS_INIT). Tag `None`.
        (None, NECROBINDER_SPECS),
        (None, COLORLESS_SPECS),
        (None, STATUS_SPECS),
    ] {
        for factory in slice {
            let spec = factory();
            if covered.contains(&spec.id) {
                continue;
            }
            merged.push((spec.id, class, *factory));
        }
    }
    merged
}

/// Public view of the deck-buildable card pool. Returned as a
/// `Vec<CardSpec>` so callers (e.g. the CLI `--deck-preset random`
/// path in `sts2-rs-c9s5`) can sample without going through name
/// resolution. Order matches [`deck_card_spec_registry`] (declaration
/// order — Ironclad starters first, then Silent starters, then
/// common demo cards). Stable enough for seeded sampling.
#[must_use]
pub fn deck_card_spec_pool() -> Vec<crate::cards::CardSpec> {
    deck_card_spec_registry()
        .into_iter()
        .map(|(_, _, f)| f())
        .collect()
}

/// Look `card_id` up in a [`deck_card_spec_pool`] snapshot, falling back to
/// the v0.111.0-only cards ([`crate::cards::spec_index::V0111_ONLY_SPECS`]).
///
/// Those cards are kept out of the pool on purpose (it feeds the CLI
/// random-deck samplers), but a v0.111.0 run can still hold one — Abundance
/// via the Dowsing quest — and every deck-card eligibility scan must see it
/// (C# `BeautifulBracelet.AfterObtained` filters the whole Deck pile by
/// `swift.CanEnchant`, BeautifulBracelet.cs:31-44). Same fallback shape as
/// [`resolve_card_name`]. The fallback is only reached on a pool miss, so it
/// cannot change a v0.107.1 result.
#[must_use]
pub fn find_deck_card_spec(
    pool: &[crate::cards::CardSpec],
    card_id: CardId,
) -> Option<crate::cards::CardSpec> {
    pool.iter().find(|s| s.id == card_id).copied().or_else(|| {
        crate::cards::spec_index::V0111_ONLY_SPECS
            .iter()
            .map(|f| f())
            .find(|spec| spec.id == card_id)
    })
}

/// Class-filtered view of [`deck_card_spec_pool`]. Used by the CLI
/// `--deck-preset random` path (bead `sts2-rs-9yoh` / cliext F4) to
/// avoid mixing Ironclad and Silent cards in a single random deck.
#[must_use]
pub fn deck_card_spec_pool_for_class(class: PlayerClassHint) -> Vec<crate::cards::CardSpec> {
    deck_card_spec_registry()
        .into_iter()
        .filter(|(_, c, _)| *c == Some(class))
        .map(|(_, _, f)| f())
        .collect()
}

/// Resolve a user-supplied card name to a [`CardSpec`].
///
/// Accepts every form `card_id_from_str` accepts, plus bare `Strike` /
/// `Defend` (resolved against `class_hint`). Returns
/// [`UnknownCard`] with up to 3 did-you-mean suggestions on miss.
///
/// # Errors
///
/// Returns [`UnknownCard`] when the input does not resolve to any
/// deck-buildable [`CardId`].
pub fn resolve_card_name(
    input: &str,
    class_hint: Option<PlayerClassHint>,
) -> Result<crate::cards::CardSpec, UnknownCard> {
    // Bare Strike/Defend → class-specific variant.
    let disambiguated = match (input.trim().to_ascii_lowercase().as_str(), class_hint) {
        ("strike", Some(PlayerClassHint::Ironclad)) => Some("StrikeIronclad".to_string()),
        ("strike", Some(PlayerClassHint::Silent)) => Some("StrikeSilent".to_string()),
        ("defend", Some(PlayerClassHint::Ironclad)) => Some("DefendIronclad".to_string()),
        ("defend", Some(PlayerClassHint::Silent)) => Some("DefendSilent".to_string()),
        _ => None,
    };
    let lookup_input = disambiguated.as_deref().unwrap_or(input);

    let card_id = card_id_from_str(lookup_input).ok_or_else(|| UnknownCard {
        input: input.to_string(),
        suggestions: suggest_card_id(input),
    })?;

    deck_card_spec_registry()
        .into_iter()
        .find_map(|(id, _class, f)| (id == card_id).then(f))
        // GH #793: a v0.111.0-only card (kept out of the v0.107.1 registry
        // the CLI pools read).
        .or_else(|| {
            crate::cards::spec_index::V0111_ONLY_SPECS
                .iter()
                .map(|f| f())
                .find(|spec| spec.id == card_id)
        })
        .ok_or_else(|| UnknownCard {
            input: input.to_string(),
            // CardId resolved but no factory registered — surface the
            // canonical form so the user can confirm the name.
            suggestions: vec![pascal_display(card_id.to_python_name())
                .split_whitespace()
                .collect()],
        })
}

/// Build a `(values, python_names)` pair for [`RelicId`].
fn relic_id_tables() -> (Vec<RelicId>, Vec<&'static str>) {
    let values: Vec<RelicId> = RelicId::all().to_vec();
    let names: Vec<&'static str> = values.iter().map(|v| v.to_python_name()).collect();
    (values, names)
}

/// Token aliases for relic ids whose sts2log name diverged from the
/// Python-pinned canonical at `01c094d`. Bead `sts2-rs-zhdg` — see
/// `docs/reality-checks/2026-05-16-sts2log-content-gap-report.md`
/// §"Unknown relic IDs".
///
/// Previously held `NEOWS_BONES → NEOWS_TORMENT`, a 2026-05-16 guess
/// that the token was an old name for `NeowsTorment`. That was wrong:
/// the authoritative C# v0.103.2 has `NeowsBones` (curse-pool Ancient
/// relic — grant 2 relics + 1 curse, `NeowsBones.cs`) and `NeowsTorment`
/// (positive-pool Ancient relic — add a `NeowsFury` card, `NeowsTorment.cs`)
/// as two *distinct* relics. Bead `sts2-rs-wjuoi` (W9e) ported
/// `RelicId::NeowsBones`, so `NEOWS_BONES` now resolves canonically to
/// its own variant and the alias was removed (a scraped `NEOWS_BONES`
/// pick is the curse-relic, not `NeowsTorment`).
const RELIC_ID_TOKEN_ALIASES: &[(&str, &str)] = &[];

/// sts2log tokens that intentionally resolve to *nothing* in the
/// lenient fixture path. Two cohorts:
///
/// 1. **Mod content** riding the relic-id namespace. Honkai-Star-Rail
///    `*_RUNE` runes and similar — these are not STS2 relics and
///    should never get a `RelicId` variant. The scraper should
///    eventually filter these upstream (`fixtures emit-sts2log
///    --skip-mod-content` is the proper fix; tracked in the gap
///    report); until then the harness drops them silently.
/// 2. **Canonical-but-unported**: currently empty. Tokens land here when
///    real STS2 content has no `RelicId` variant yet.
///
///    `WINGED_BOOTS`, `NEOWS_TALISMAN`, `PHIAL_HOLSTER` (bead
///    `sts2-rs-4rwv4`, W9a-c), `NEOWS_BONES` (bead `sts2-rs-wjuoi`,
///    W9e), `HEFTY_TABLET` (bead `sts2-rs-w6hdj`, W9d), and the three
///    v0.107.1 Ancient/Neow relics `SILKEN_TRESS` / `FISHING_ROD` /
///    `KALEIDOSCOPE` (bead `sts2-rs-v0107-content-port-wb8lk.6`, appended
///    to `relic_ids_generated.rs` as C#-only extras) *were* in
///    this cohort but now resolve to real `RelicId` variants, so they
///    were removed from the skip list.
///
/// Bead `sts2-rs-zhdg`.
const LENIENT_SKIP_RELIC_TOKENS: &[&str] = &[
    // Mod content (Honkai etc.) — filter upstream eventually.
    "WATCH_OUT_GRAPEFRUIT_RUNE",
    "SINGULARITY_AI_RUNE",
    // NB: `SILKEN_TRESS` and `FISHING_ROD` *were* here as
    // canonical-but-unported; bead `sts2-rs-v0107-content-port-wb8lk.6`
    // ported their `RelicId` variants (plus `KALEIDOSCOPE`), so they now
    // resolve canonically and left this skip list.
];

/// sts2log card tokens that intentionally resolve to *nothing* in the
/// lenient fixture path — same shape as [`LENIENT_SKIP_RELIC_TOKENS`]
/// but for cards. Currently a tiny set covering Python-pin gaps that
/// the scraper still emits.
///
/// Bead `sts2-rs-qt0b` (fwxe.6): `NOT_YET` is real STS2 content (see
/// C# at `MegaCrit.Sts2.Core.Models.Cards/NotYet.cs`) but absent from
/// the Python pin at `01c094d`, so `CardId` (auto-generated) lacks the
/// variant. Mirrors the SILKEN_TRESS/HEFTY_TABLET pattern from
/// `LENIENT_SKIP_RELIC_TOKENS`. Add tokens here when the scraper
/// surfaces another canonical-but-unported card.
const LENIENT_SKIP_CARD_TOKENS: &[&str] = &[
    // Canonical STS2 but missing from Python pin 01c094d.
    "NOT_YET",
];

/// True iff `token` is a documented sts2log card token that resolves
/// to nothing in the lenient fixture path. Bead `sts2-rs-qt0b`.
#[must_use]
pub fn is_lenient_skip_card_token(token: &str) -> bool {
    let upper = token.trim().to_ascii_uppercase();
    LENIENT_SKIP_CARD_TOKENS
        .iter()
        .any(|t| t.eq_ignore_ascii_case(upper.as_str()))
}

/// True iff `token` is a documented sts2log token that resolves to
/// nothing in the lenient harness path. `*_RUNE`-suffixed tokens are
/// treated as mod content categorically, even if not explicitly
/// enumerated — this drains future variants of the same pattern.
#[must_use]
pub fn is_lenient_skip_relic_token(token: &str) -> bool {
    let upper = token.trim().to_ascii_uppercase();
    if upper.ends_with("_RUNE") {
        return true;
    }
    LENIENT_SKIP_RELIC_TOKENS
        .iter()
        .any(|t| t.eq_ignore_ascii_case(upper.as_str()))
}

/// Resolve a CLI/scenario string to a [`RelicId`].
///
/// Accepts:
/// - `PascalCase` / human-display / snake_case via [`name_lookup`].
/// - sts2log historical aliases listed in [`RELIC_ID_TOKEN_ALIASES`]
///   (currently empty — see that const's doc-comment).
#[must_use]
pub fn relic_id_from_str(input: &str) -> Option<RelicId> {
    let (values, names) = relic_id_tables();
    if let Some(hit) = name_lookup(input, &values, &names) {
        return Some(hit);
    }
    let normalized = normalize(input);
    for (from, to) in RELIC_ID_TOKEN_ALIASES {
        if normalize(from) == normalized {
            return name_lookup(to, &values, &names);
        }
    }
    None
}

/// Top-3 did-you-mean suggestions for an unresolved [`RelicId`] input.
#[must_use]
pub fn suggest_relic_id(input: &str) -> Vec<String> {
    let (_, names) = relic_id_tables();
    suggest(input, &names)
}

/// Build a `(values, python_names)` pair for [`PotionKind`].
fn potion_kind_tables() -> (Vec<PotionKind>, Vec<&'static str>) {
    let values: Vec<PotionKind> = PotionKind::ALL.to_vec();
    let names: Vec<&'static str> = values.iter().map(|v| v.python_name()).collect();
    (values, names)
}

/// Resolve a CLI/scenario string to a [`PotionKind`].
#[must_use]
pub fn potion_kind_from_str(input: &str) -> Option<PotionKind> {
    let (values, names) = potion_kind_tables();
    name_lookup(input, &values, &names)
}

/// Top-3 did-you-mean suggestions for an unresolved [`PotionKind`]
/// input.
#[must_use]
pub fn suggest_potion_kind(input: &str) -> Vec<String> {
    let (_, names) = potion_kind_tables();
    suggest(input, &names)
}

/// Render a `PascalCase` variant name as a human display string by
/// inserting a space before every uppercase-after-lowercase boundary.
/// Exposed for CLI banner formatting (plan §3.3 / §3.6).
#[must_use]
pub fn pascal_display(pascal: &str) -> String {
    pascal_to_display(pascal)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------- levenshtein arithmetic --------

    #[test]
    fn levenshtein_exact_distances() {
        // Identical strings: 0.
        assert_eq!(levenshtein("burning", "burning"), 0);
        assert_eq!(levenshtein("", ""), 0);
        // Pure insertions from empty: distance == length of the other.
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        // Single substitution: 1 (catches `prev[j-1] + cost`).
        assert_eq!(levenshtein("cat", "bat"), 1);
        // Single insertion / deletion: 1 (catches `prev[j] + 1`,
        // `curr[j-1] + 1`).
        assert_eq!(levenshtein("cat", "cart"), 1);
        assert_eq!(levenshtein("cart", "cat"), 1);
        // Classic mixed case: 3 (2 substitutions + 1 insertion).
        // Pins the `+`/`-` arithmetic in the inner recurrence — a `+`→`*`
        // or `-`→`/` mutation changes this number.
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        // Two substitutions.
        assert_eq!(levenshtein("abc", "xyz"), 3);
    }

    // -------- PlayerClassHint::as_tag / from_tag --------

    #[test]
    fn player_class_hint_as_tag_exact_strings() {
        assert_eq!(PlayerClassHint::Ironclad.as_tag(), "ironclad");
        assert_eq!(PlayerClassHint::Silent.as_tag(), "silent");
        assert_eq!(PlayerClassHint::Defect.as_tag(), "defect");
    }

    #[test]
    fn player_class_hint_from_tag_known_and_unknown() {
        assert_eq!(
            PlayerClassHint::from_tag("ironclad"),
            Some(PlayerClassHint::Ironclad)
        );
        assert_eq!(
            PlayerClassHint::from_tag("silent"),
            Some(PlayerClassHint::Silent)
        );
        assert_eq!(
            PlayerClassHint::from_tag("defect"),
            Some(PlayerClassHint::Defect)
        );
        // Case-insensitive + trimmed (the body lowercases and trims).
        assert_eq!(
            PlayerClassHint::from_tag("  IRONCLAD  "),
            Some(PlayerClassHint::Ironclad)
        );
        // Unknown tag → None.
        assert_eq!(PlayerClassHint::from_tag("not_a_class"), None);
        assert_eq!(PlayerClassHint::from_tag(""), None);
    }

    #[test]
    fn player_class_hint_tag_round_trips_every_variant() {
        for variant in [
            PlayerClassHint::Ironclad,
            PlayerClassHint::Silent,
            PlayerClassHint::Defect,
        ] {
            assert_eq!(
                PlayerClassHint::from_tag(variant.as_tag()),
                Some(variant),
                "round-trip {variant:?}"
            );
        }
    }

    // -------- normalize / snake_to_pascal / pascal_to_display --------

    // -------- cliext H4 registry merge --------

    #[test]
    fn deck_card_spec_pool_contains_bloodletting_and_other_generated_cards() {
        // H4 acceptance: cards that have a generated factory should
        // resolve via the registry even when they are not in the
        // curated demo list.
        let pool = deck_card_spec_pool();
        let ids: std::collections::HashSet<_> = pool.iter().map(|s| s.id).collect();
        for id in [
            CardId::Bloodletting, // ironclad
            CardId::BloodWall,
            CardId::Hemokinesis,
            // silent
            CardId::Adrenaline,
            CardId::Backflip,
            // defect (new for H4)
            CardId::Capacitor,
            CardId::IceLance,
        ] {
            assert!(
                ids.contains(&id),
                "expected {id:?} in deck pool after H4 merge"
            );
        }
    }

    #[test]
    fn deck_card_spec_pool_for_class_disjoint_across_classes() {
        // Class-locked pools must not leak across classes — this is
        // what the random-deck class lock relies on (H4 + F4).
        let ironclad: std::collections::HashSet<_> =
            deck_card_spec_pool_for_class(PlayerClassHint::Ironclad)
                .iter()
                .map(|s| s.id)
                .collect();
        let silent: std::collections::HashSet<_> =
            deck_card_spec_pool_for_class(PlayerClassHint::Silent)
                .iter()
                .map(|s| s.id)
                .collect();
        let defect: std::collections::HashSet<_> =
            deck_card_spec_pool_for_class(PlayerClassHint::Defect)
                .iter()
                .map(|s| s.id)
                .collect();
        assert!(ironclad.is_disjoint(&silent));
        assert!(ironclad.is_disjoint(&defect));
        assert!(silent.is_disjoint(&defect));
        // Sanity: each class must have enough cards for a random
        // deck of size 10 to be feasible.
        for (name, set) in [
            ("ironclad", &ironclad),
            ("silent", &silent),
            ("defect", &defect),
        ] {
            assert!(set.len() >= 10, "{name} pool too small: {}", set.len());
        }
    }

    #[test]
    fn deck_pool_size_meets_h4_expectation() {
        // Curated 30 + generated 354 = ~384, modulo CardId collisions
        // (curated wins). Accept a band so future regens don't break
        // the test; the floor matters more than the ceiling.
        let n = deck_card_spec_pool().len();
        assert!(n >= 350, "deck pool too small after H4 merge: {n}");
        assert!(
            n < 600,
            "deck pool exceeds CardId::all().len() — duplicates leaked: {n}"
        );
    }

    #[test]
    fn normalize_collapses_all_four_input_forms() {
        let canon = "burningblood";
        assert_eq!(normalize("BurningBlood"), canon);
        assert_eq!(normalize("Burning Blood"), canon);
        assert_eq!(normalize("burning blood"), canon);
        assert_eq!(normalize("burning_blood"), canon);
        assert_eq!(normalize("BURNING_BLOOD"), canon);
    }

    #[test]
    fn snake_to_pascal_round_trips_common_names() {
        assert_eq!(snake_to_pascal("BURNING_BLOOD"), "BurningBlood");
        assert_eq!(snake_to_pascal("RING_OF_THE_SNAKE"), "RingOfTheSnake");
        assert_eq!(snake_to_pascal("BASH"), "Bash");
    }

    #[test]
    fn pascal_to_display_inserts_spaces() {
        assert_eq!(pascal_to_display("BurningBlood"), "Burning Blood");
        assert_eq!(pascal_to_display("RingOfTheSnake"), "Ring Of The Snake");
        assert_eq!(pascal_to_display("Bash"), "Bash");
    }

    // -------- RelicId acceptance (verbatim from bead sts2-rs-e5j9) --------

    #[test]
    fn relic_id_dual_form_lookup() {
        assert_eq!(
            relic_id_from_str("BurningBlood"),
            Some(RelicId::BurningBlood)
        );
        assert_eq!(
            relic_id_from_str("Burning Blood"),
            Some(RelicId::BurningBlood)
        );
        assert_eq!(
            relic_id_from_str("burning blood"),
            Some(RelicId::BurningBlood)
        );
        assert_eq!(
            relic_id_from_str("burning_blood"),
            Some(RelicId::BurningBlood)
        );
        assert_eq!(relic_id_from_str("bogus"), None);
        assert_eq!(relic_id_from_str(""), None);
    }

    #[test]
    fn relic_id_suggest_finds_close_typos() {
        let suggestions = suggest_relic_id("burningbleed");
        assert!(
            suggestions.iter().any(|s| s == "BurningBlood"),
            "expected BurningBlood in top-3 suggestions, got {suggestions:?}"
        );
        assert!(suggestions.len() <= MAX_SUGGESTIONS);
    }

    // -------- CardId acceptance --------

    #[test]
    fn card_id_dual_form_lookup() {
        // `IronWave` is a P6 Ironclad card with a multi-word PascalCase
        // variant — exercises space-vs-no-space resolution.
        assert_eq!(card_id_from_str("IronWave"), Some(CardId::IronWave));
        assert_eq!(card_id_from_str("Iron Wave"), Some(CardId::IronWave));
        assert_eq!(card_id_from_str("iron wave"), Some(CardId::IronWave));
        assert_eq!(card_id_from_str("iron_wave"), Some(CardId::IronWave));
        // Single-word variant.
        assert_eq!(card_id_from_str("Bash"), Some(CardId::Bash));
        assert_eq!(card_id_from_str("bash"), Some(CardId::Bash));
        assert_eq!(card_id_from_str("bogus"), None);
    }

    #[test]
    fn card_id_suggest_finds_close_typos() {
        let suggestions = suggest_card_id("ironwave");
        assert!(
            suggestions.iter().any(|s| s == "IronWave"),
            "expected IronWave in top-3 suggestions, got {suggestions:?}"
        );
    }

    // -------- PotionKind acceptance --------

    #[test]
    fn potion_kind_dual_form_lookup() {
        assert_eq!(
            potion_kind_from_str("FirePotion"),
            Some(PotionKind::FirePotion)
        );
        assert_eq!(
            potion_kind_from_str("Fire Potion"),
            Some(PotionKind::FirePotion)
        );
        assert_eq!(
            potion_kind_from_str("fire potion"),
            Some(PotionKind::FirePotion)
        );
        assert_eq!(
            potion_kind_from_str("fire_potion"),
            Some(PotionKind::FirePotion)
        );
        assert_eq!(
            potion_kind_from_str("BloodPotion"),
            Some(PotionKind::BloodPotion)
        );
        assert_eq!(potion_kind_from_str("bogus"), None);
    }

    #[test]
    fn potion_kind_suggest_finds_close_typos() {
        let suggestions = suggest_potion_kind("firepotian");
        assert!(
            suggestions.iter().any(|s| s == "FirePotion"),
            "expected FirePotion in top-3 suggestions, got {suggestions:?}"
        );
    }

    // -------- Sanity: every variant is round-trippable. --------

    #[test]
    fn every_card_id_resolves_from_its_pascal_form() {
        for &id in CardId::all() {
            let pascal = snake_to_pascal(id.to_python_name());
            assert_eq!(card_id_from_str(&pascal), Some(id), "round-trip {pascal}");
        }
    }

    #[test]
    fn every_relic_id_resolves_from_its_pascal_form() {
        for &id in RelicId::all() {
            let pascal = snake_to_pascal(id.to_python_name());
            assert_eq!(relic_id_from_str(&pascal), Some(id), "round-trip {pascal}");
        }
    }

    #[test]
    fn every_potion_kind_resolves_from_its_pascal_form() {
        for &id in &PotionKind::ALL {
            let pascal = snake_to_pascal(id.python_name());
            assert_eq!(
                potion_kind_from_str(&pascal),
                Some(id),
                "round-trip {pascal}"
            );
        }
    }

    // ── _CARD-suffix-stripped CardId resolution (bead sts2-rs-u5zc) ─────

    #[test]
    fn card_id_from_str_resolves_v0113_only_cards() {
        assert_eq!(card_id_from_str("MY_SHINIES"), Some(CardId::MyShinies));
        assert_eq!(card_id_from_str("DINGY_DAGGER"), Some(CardId::DingyDagger));
    }

    #[test]
    fn card_id_from_str_accepts_rupture_without_suffix() {
        assert_eq!(card_id_from_str("RUPTURE"), Some(CardId::RuptureCard));
    }

    #[test]
    fn card_id_from_str_still_accepts_canonical_rupture_card() {
        assert_eq!(card_id_from_str("RUPTURE_CARD"), Some(CardId::RuptureCard));
        assert_eq!(card_id_from_str("Rupture"), Some(CardId::RuptureCard));
        assert_eq!(card_id_from_str("RuptureCard"), Some(CardId::RuptureCard));
    }

    #[test]
    fn card_id_from_str_drains_sts2log_gap_report_cards() {
        // Per docs/reality-checks/2026-05-16-sts2log-content-gap-report.md
        // §72, these 9 cards block 80 fixtures purely due to the
        // suffix mismatch. Each must resolve via the _CARD-stripped
        // fallback. All have wired bodies in the Ironclad pool.
        let cases = [
            ("RUPTURE", CardId::RuptureCard),
            ("VICIOUS", CardId::ViciousCard),
            ("INFERNO", CardId::InfernoCard),
            ("RAGE", CardId::RageCard),
            ("BARRICADE", CardId::BarricadeCard),
            // SPLASH has no `_CARD` suffix in the Rust enum
            // (`CardId::Splash`), so it resolves via the first lookup
            // pass rather than the suffix-strip fallback; included
            // here so the gap-report's 9 cards all carry a coverage
            // assertion.
            ("SPLASH", CardId::Splash),
            ("JUGGLING", CardId::JugglingCard),
            ("SETUP_STRIKE", CardId::SetupStrikeCard),
            ("STAMPEDE", CardId::StampedeCard),
        ];
        for (token, expected) in cases {
            assert_eq!(
                card_id_from_str(token),
                Some(expected),
                "sts2log token {token} must resolve",
            );
        }
    }

    #[test]
    fn relic_id_from_str_drains_sts2log_gap_report_relic_ids() {
        // Per docs/reality-checks/2026-05-16-sts2log-content-gap-report.md
        // §"Unknown relic IDs" (22 blocked fixtures, 6 named tokens +
        // 2 "other"). Each entry below MUST hit one of three
        // dispositions:
        //
        // - canonical: token resolves directly to a `RelicId` variant
        //   (e.g. NEOWS_BONES → NeowsBones, ported in W9e).
        // - lenient-skip: documented mod-content or
        //   canonical-but-unported; resolver returns None and the
        //   fixture loader drops the relic silently.
        // - canonical: token already resolves through the primary
        //   `relic_id_from_str` pass (unlikely for the gap report —
        //   those tokens are by-definition missing — but included for
        //   forward-compatibility once Python-pin bumps).
        //
        // Bead `sts2-rs-zhdg`.
        enum Disposition {
            Canonical(RelicId),
            LenientSkip,
        }
        use Disposition::*;
        let cases: &[(&str, Disposition)] = &[
            // Bead `sts2-rs-wjuoi` (W9e): NEOWS_BONES is now its own
            // distinct RelicId (was wrongly aliased to NeowsTorment).
            ("NEOWS_BONES", Canonical(RelicId::NeowsBones)),
            // Bead `sts2-rs-w6hdj` (W9d): HEFTY_TABLET now ports to its
            // own RelicId (3-Rare select + Injury), so it resolves
            // canonically instead of lenient-skipping.
            ("HEFTY_TABLET", Canonical(RelicId::HeftyTablet)),
            ("WATCH_OUT_GRAPEFRUIT_RUNE", LenientSkip),
            ("SINGULARITY_AI_RUNE", LenientSkip),
            // Bead `sts2-rs-v0107-content-port-wb8lk.6`: the three v0.107.1
            // Ancient/Neow relics now port to their own RelicId variants, so
            // they resolve canonically instead of lenient-skipping.
            ("SILKEN_TRESS", Canonical(RelicId::SilkenTress)),
            ("FISHING_ROD", Canonical(RelicId::FishingRod)),
            ("KALEIDOSCOPE", Canonical(RelicId::Kaleidoscope)),
        ];
        for (token, disp) in cases {
            match disp {
                Canonical(expected) => {
                    assert_eq!(
                        relic_id_from_str(token),
                        Some(*expected),
                        "sts2log token {token} must resolve canonically",
                    );
                    assert!(
                        !is_lenient_skip_relic_token(token),
                        "canonical token {token} must NOT be in the lenient-skip list",
                    );
                }
                LenientSkip => {
                    assert_eq!(
                        relic_id_from_str(token),
                        None,
                        "sts2log token {token} must NOT resolve to a RelicId variant",
                    );
                    assert!(
                        is_lenient_skip_relic_token(token),
                        "sts2log token {token} must be flagged for lenient skip",
                    );
                }
            }
        }
        // The general `*_RUNE` pattern is the catch-all for unenumerated
        // mod-content runes; pin it explicitly so future scraped
        // anomalies stay in the skip path without needing a memo bump.
        assert!(is_lenient_skip_relic_token("ANY_FUTURE_MOD_RUNE"));
        assert!(is_lenient_skip_relic_token("another_future_rune"));
        // Real canonical tokens must NOT be lenient-skipped.
        assert!(!is_lenient_skip_relic_token("BurningBlood"));
        assert!(!is_lenient_skip_relic_token("BURNING_BLOOD"));
        // Bead `sts2-rs-4rwv4` (W9a-c): these three left the skip list
        // once their `RelicId` variants landed — they must now resolve
        // and must NOT be lenient-skipped.
        for token in ["WINGED_BOOTS", "NEOWS_TALISMAN", "PHIAL_HOLSTER"] {
            assert!(
                relic_id_from_str(token).is_some(),
                "W9 token {token} must now resolve to a RelicId variant",
            );
            assert!(
                !is_lenient_skip_relic_token(token),
                "W9 token {token} must NOT be lenient-skipped after porting",
            );
        }
    }

    /// Bead `sts2-rs-2vze` — name resolution alone is necessary but not
    /// sufficient. The deck-builder must also be able to construct each
    /// of the 9 gap-report cards as a `CardSpec` (i.e. have a factory
    /// registered in `deck_card_spec_registry`), otherwise the sts2log
    /// replay path errors at deck-seed time even though the token
    /// parsed. Asserts the full path `token → resolve_card_name →
    /// CardSpec` works for all 9. Pair of `..._drains_..._cards` above.
    #[test]
    fn resolve_card_name_drains_sts2log_gap_report_cards() {
        let cards = [
            ("RUPTURE", CardId::RuptureCard),
            ("VICIOUS", CardId::ViciousCard),
            ("INFERNO", CardId::InfernoCard),
            ("RAGE", CardId::RageCard),
            ("BARRICADE", CardId::BarricadeCard),
            ("SPLASH", CardId::Splash),
            ("JUGGLING", CardId::JugglingCard),
            ("SETUP_STRIKE", CardId::SetupStrikeCard),
            ("STAMPEDE", CardId::StampedeCard),
            // Bead `sts2-rs-gkpd` (nmy9 follow-up): three colorless
            // factories added to COLORLESS_SPECS in this commit.
            ("THE_BOMB", CardId::TheBombCard),
            ("VOLLEY", CardId::Volley),
            ("THRUMMING_HATCHET", CardId::ThrummingHatchet),
        ];
        for (token, expected_id) in cards {
            let spec = resolve_card_name(token, None).unwrap_or_else(|e| {
                panic!("sts2log token {token} must resolve to a CardSpec: {e:?}")
            });
            assert_eq!(
                spec.id, expected_id,
                "{token} resolved to wrong CardId: got {:?}",
                spec.id,
            );
        }
    }

    #[test]
    fn resolve_card_name_loads_all_b3jbd_colorless_cards() {
        // Bead `sts2-rs-b3jbd`: these 29 colorless cards have generated
        // `CardSpec` factories in `.rs` and
        // resolve a `CardId` (their names are in `card_id_tables`), but
        // were never listed in `COLORLESS_SPECS`, so `resolve_card_name`'s
        // second-stage `CardId -> CardSpec` lookup failed with
        // `UnknownCard`. Surfaced by the 2026-05-30 exhaustive parity
        // coverage sweep (every one bucketed HARNESS_INIT). C# loads all
        // 29 fine — a pure Rust deck-loader registration gap. Each must
        // now materialize a deck spec.
        let tokens = [
            "ANOINTED",
            "BEACON_OF_HOPE",
            "BEAT_DOWN",
            "BELIEVE_IN_YOU",
            "BOLAS",
            "CALAMITY_CARD",
            "CATASTROPHE",
            "DARK_SHACKLES",
            "EQUILIBRIUM",
            "ETERNAL_ARMOR",
            "GANG_UP",
            "GOLD_AXE",
            "IMPATIENCE",
            "INTERCEPT_CARD",
            "JACK_OF_ALL_TRADES",
            "KNOCKDOWN",
            "LIFT",
            "MAYHEM_CARD",
            "MIMIC",
            "MIND_BLAST",
            "NOSTALGIA_CARD",
            "PROLONG",
            "REND",
            "RESTLESSNESS",
            "ROLLING_BOULDER",
            "SALVO",
            "SEEKER_STRIKE",
            "STRATAGEM",
            "TAG_TEAM",
        ];
        for token in tokens {
            let spec = resolve_card_name(token, None)
                .unwrap_or_else(|e| panic!("colorless card {token} must deck-load: {e:?}"));
            assert_eq!(
                spec.name, token,
                "{token} resolved to a spec with mismatched name {:?}",
                spec.name,
            );
        }
    }

    #[test]
    fn resolve_card_name_loads_all_necrobinder_cards() {
        // Same class of gap as `..._b3jbd_colorless_cards`: every
        // `NECROBINDER_SPECS` card has a generated `CardSpec` factory and a
        // resolvable `CardId`, but `NECROBINDER_SPECS` was never merged into
        // `deck_card_spec_registry`, so `resolve_card_name`'s `CardId ->
        // CardSpec` lookup failed with `UnknownCard` for ALL 86 Necrobinder
        // cards. Surfaced by the per-character parity coverage / v2 sweeps
        // (every Necrobinder trial bucketed HARNESS_INIT). Iterate the slice
        // directly so this can't drift as cards are added.
        use crate::cards::spec_index::NECROBINDER_SPECS;
        for factory in NECROBINDER_SPECS {
            let name = factory().name;
            let spec = resolve_card_name(name, None)
                .unwrap_or_else(|e| panic!("necrobinder card {name} must deck-load: {e:?}"));
            assert_eq!(
                spec.name, name,
                "{name} resolved to a spec with mismatched name {:?}",
                spec.name,
            );
        }
    }

    // -------- resolve_card_name bare Strike/Defend disambiguation --------

    /// Each `(token, class_hint)` arm must resolve to that class's variant.
    /// Deleting any arm drops to `_ => None`, after which `card_id_from_str`
    /// cannot resolve bare "strike"/"defend" and the call returns
    /// `UnknownCard` — so asserting the resolved id kills every arm deletion.
    #[test]
    fn resolve_card_name_disambiguates_bare_strike_and_defend_by_class() {
        let cases = [
            ("strike", PlayerClassHint::Ironclad, CardId::StrikeIronclad),
            ("strike", PlayerClassHint::Silent, CardId::StrikeSilent),
            ("defend", PlayerClassHint::Ironclad, CardId::DefendIronclad),
            ("defend", PlayerClassHint::Silent, CardId::DefendSilent),
        ];
        for (token, hint, expected) in cases {
            let spec = resolve_card_name(token, Some(hint))
                .unwrap_or_else(|e| panic!("{token}/{hint:?} must resolve: {e:?}"));
            assert_eq!(
                spec.id, expected,
                "bare {token:?} with {hint:?} must resolve to {expected:?}"
            );
        }
    }

    // -------- is_lenient_skip_card_token --------

    /// Pins both the true (documented skip token) and false (real card)
    /// branches so the `-> true` / `-> false` constant replacements both die.
    #[test]
    fn is_lenient_skip_card_token_only_for_documented_tokens() {
        assert!(
            is_lenient_skip_card_token("NOT_YET"),
            "NOT_YET is a documented lenient-skip token"
        );
        assert!(
            !is_lenient_skip_card_token("STRIKE_IRONCLAD"),
            "a real card token must not be lenient-skipped"
        );
    }

    // -------- pascal_display public wrapper --------

    /// The public `pascal_display` wrapper must insert word boundaries;
    /// pins a concrete non-empty rendering so `-> String::new()` and
    /// `-> "xyzzy"` both fail.
    #[test]
    fn pascal_display_wrapper_inserts_word_boundaries() {
        assert_eq!(pascal_display("PascalCase"), "Pascal Case");
        assert_eq!(pascal_display("Strike"), "Strike");
    }
}
