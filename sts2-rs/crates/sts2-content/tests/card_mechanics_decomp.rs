//! `sts2_core::card_mechanics` against the decompiled cards: the four card
//! sets it describes are exactly the decomp's.
//!
//! - channels an orb: the card's `OnPlay` calls `OrbCmd.Channel`;
//! - summons Osty: it calls `OstyCmd.Summon`;
//! - needs a living Osty: it calls `Osty.CheckMissingWithAnim`;
//! - works the orb belt otherwise: it evokes, passives, adds or removes
//!   slots, reads `OrbQueue`, touches `FocusPower`, or applies a Power whose
//!   own code calls `OrbCmd` or reads `OrbQueue` (`Loop`, `Storm`, …).
//!
//! A card that gains or loses one of these effects in a new build fails here
//! instead of reaching the AI's features silently wrong.
//!
//! Reads `STS2_DECOMP_DIR` (the `v0.107.1` root) or the sibling `sts2-wiki`
//! checkout's v0.107.1 decomp; skips (with a note) when neither exists.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use sts2_content::names::card_id_from_str;
use sts2_core::card_mechanics::card_mechanics;
use sts2_core::cards::CardId;

/// The decomp's model directory for `kind` (`Cards`, `Powers`): the one
/// namespace directory under `root` named `<...>.Models.<kind>`.
fn models(root: &Path, kind: &str) -> PathBuf {
    let suffix = format!(".Models.{kind}");
    std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with(&suffix))
        })
        .unwrap_or_else(|| root.join(&suffix[1..]))
}

/// The pinned game build whose decompilation the table was transcribed from.
const DECOMP_VERSION: &str = "v0.107.1";

fn decomp_root() -> Option<PathBuf> {
    let candidates = [
        std::env::var_os("STS2_DECOMP_DIR").map(PathBuf::from),
        Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../sts2-wiki/decompiled")
                .join(DECOMP_VERSION),
        ),
    ];
    candidates
        .into_iter()
        .flatten()
        .find(|p| models(p, "Cards").is_dir())
}

/// `(class name, source)` of every `.cs` file in `dir`.
fn sources(dir: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).expect("decomp dir").flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("cs") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_owned();
        let src = std::fs::read_to_string(&path).expect("read .cs");
        out.push((stem, src));
    }
    out.sort();
    out
}

/// The engine card each decomp class resolves to (mocks and abstract bases
/// resolve to none and are skipped).
fn cards_where(cards: &[(String, String)], pred: impl Fn(&str) -> bool) -> BTreeSet<CardId> {
    cards
        .iter()
        .filter(|(_, src)| pred(src))
        .filter_map(|(class, _)| card_id_from_str(class))
        .collect()
}

fn engine_cards_where(pred: impl Fn(CardId) -> bool) -> BTreeSet<CardId> {
    CardId::all()
        .iter()
        .copied()
        .filter(|&id| pred(id))
        .collect()
}

#[test]
fn card_mechanics_sets_match_the_decomp() {
    let Some(root) = decomp_root() else {
        eprintln!("skipping: no decompiled cards (set STS2_DECOMP_DIR)");
        return;
    };
    let cards = sources(&models(&root, "Cards"));
    assert!(
        cards.len() > 400,
        "expected the full card set, got {}",
        cards.len()
    );
    let orb_powers: Vec<String> = sources(&models(&root, "Powers"))
        .into_iter()
        .filter(|(_, src)| src.contains("OrbCmd.") || src.contains("OrbQueue"))
        .map(|(class, _)| class)
        .collect();
    assert!(
        orb_powers.iter().any(|p| p == "LoopPower"),
        "orb powers: {orb_powers:?}"
    );

    let either = |id: CardId, f: &dyn Fn(sts2_core::card_mechanics::CardMechanics) -> bool| {
        f(card_mechanics(id, false)) || f(card_mechanics(id, true))
    };

    let checks: [(&str, BTreeSet<CardId>, BTreeSet<CardId>); 4] = [
        (
            "channels an orb",
            cards_where(&cards, |s| s.contains("OrbCmd.Channel")),
            engine_cards_where(|id| either(id, &|m| m.channels_any())),
        ),
        (
            "summons Osty",
            cards_where(&cards, |s| s.contains("OstyCmd.Summon")),
            engine_cards_where(|id| either(id, &|m| m.summon > 0)),
        ),
        (
            "needs a living Osty",
            cards_where(&cards, |s| s.contains("CheckMissingWithAnim")),
            engine_cards_where(|id| either(id, &|m| m.needs_osty)),
        ),
        (
            "works the orb belt",
            cards_where(&cards, |s| {
                [
                    "OrbCmd.Evoke",
                    "OrbCmd.Passive",
                    "OrbCmd.AddSlots",
                    "OrbCmd.RemoveSlots",
                    "OrbQueue",
                    "FocusPower",
                ]
                .iter()
                .any(|p| s.contains(p))
                    || orb_powers
                        .iter()
                        .any(|p| s.contains(&format!("Apply<{p}>")))
            }),
            engine_cards_where(|id| either(id, &|m| m.orb_interaction)),
        ),
    ];
    let mut failures = Vec::new();
    for (what, decomp, engine) in &checks {
        assert!(
            !decomp.is_empty(),
            "no decomp card {what}: is the pattern stale?"
        );
        let missing: Vec<_> = decomp.difference(engine).collect();
        let extra: Vec<_> = engine.difference(decomp).collect();
        if !missing.is_empty() || !extra.is_empty() {
            failures.push(format!(
                "{what}: in the decomp only {missing:?}; in the table only {extra:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
