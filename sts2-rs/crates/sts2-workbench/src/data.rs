//! The workbench's embedded runtime data.
//!
//! The Python workbench reads its display data from the filesystem at
//! startup — the vendored `sts2-pycli/localization_eng/*.json` dumps, the
//! sibling wiki's `sts2-wiki/data/v0.107.1/` manifests and the art manifest
//! `scripts/build_asset_manifest.py` derives from the extracted art tree.
//! In the browser there is no filesystem, so this crate `include_str!`s a
//! bundle of exactly those inputs instead, produced by
//! `sts2-pycli/scripts/bundle_workbench_data.py` (`just workbench-data`;
//! `just check-workbench-data` fails on drift).
//!
//! The game-derived files live under `crates/sts2-workbench/data/game/`
//! (extracted from a game install — never published; `just import-game`
//! regenerates them) and are staged by `build.rs`, which substitutes empty
//! stubs when the directory is absent so the crate always compiles. Tests
//! that assert on real game text are gated on `cfg(workbench_game_data)`.
//! Only the curated `data/card_tags.json` is unconditionally committed.
//!
//! Layout of `crates/sts2-workbench/data/game/`:
//!
//! * `localization_eng/*.json` — byte-for-byte copies of the game's flat
//!   `{"KEY.field": "string"}` localization dumps (plus `BUILD.txt`, the
//!   game build they were vendored from).
//! * `wiki_cards.json` / `wiki_monsters.json` / `wiki_owned.json` — the wiki
//!   manifests trimmed to the fields the Python loaders read
//!   (`localization.load_card_localization` / `load_card_meta` /
//!   `load_monster_hp_ranges` / `load_owned_localization`).
//! * `asset_manifest.json` — the sorted list of art (+ audio) paths that
//!   exist in the art store; [`crate::assets::AssetResolver`] runs in
//!   manifest mode against it.
//! * `creature_scales.json` — per-creature authored scene scale + Bounds box.
//!
//! One bundle per game build (GH #801): the layout above is the DEFAULT
//! build's (v0.107.1); every other [`GameVersion`] has a full sibling set
//! under `data/game/<ver>/` (`data/game/v0.111.0/`), built by
//! `bundle_workbench_data.py` from that build's vendored loc
//! (`localization_eng_<ver>/`) and wiki aggregates. The art tree is shared,
//! so a card new in a later build resolves to the placeholder art.
//!
//! Everything is parsed once per build, lazily, into [`Data`] (see
//! [`data_for`]; [`data`] is the default build's).

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use sts2_core::GameVersion;

/// The DEFAULT game build's wiki version — what [`data`] describes.
/// Mirrors `workbench.localization._ENGINE_WIKI_VERSION`; every build's is
/// [`engine_wiki_version`].
pub const ENGINE_WIKI_VERSION: &str = "v0.107.1";

/// The wiki `data/<ver>/` version (and vendored `BUILD.txt` line) that
/// build `version`'s bundle describes. Mirrors
/// `workbench.localization.engine_wiki_version`.
#[must_use]
pub const fn engine_wiki_version(version: GameVersion) -> &'static str {
    match version {
        GameVersion::V0_107_1 => ENGINE_WIKI_VERSION,
        GameVersion::V0_111_0 => "v0.111.0",
        GameVersion::V0_113_0 => "v0.113.0",
    }
}

/// A flat `{"KEY.field": "string"}` localization dump.
pub type LocMap = BTreeMap<String, String>;

/// One creature's authored scene scale + Bounds box
/// (`{"scale", "bounds_w", "bounds_h"}`; a bound can be `null`).
///
/// Kept as a map rather than a struct so the JSON round-trips with exactly
/// the keys the file carries — the Python resolver hands the dict through
/// untouched and the frontend reads it.
pub type CreatureScale = BTreeMap<String, Option<f64>>;

/// One keyword-only upgrade delta of a wiki card (`upgrades[]` entry with a
/// `keyword`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct WikiCardUpgrade {
    #[serde(default)]
    pub keyword: Option<String>,
    #[serde(default)]
    pub action: Option<String>,
}

/// One template variable of a wiki card (`vars[]` entry).
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct WikiCardVar {
    #[serde(rename = "type", default)]
    pub var_type: Option<String>,
    #[serde(default)]
    pub base_value: Option<i64>,
    #[serde(default)]
    pub upgraded_value: Option<i64>,
}

/// A wiki `cards.json` entry, trimmed to the fields the loaders read.
///
/// Every field but `loc_key` is optional and defaults to absent, so a
/// hand-written test fixture can be as sparse as the Python ones — the
/// loaders apply the same truthiness rules as the Python code.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct WikiCard {
    #[serde(default)]
    pub loc_key: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub character: Option<String>,
    #[serde(default)]
    pub star_cost: Option<i64>,
    #[serde(default)]
    pub x_star_cost: Option<bool>,
    #[serde(default)]
    pub description_template: Option<String>,
    #[serde(default)]
    pub description_plain: Option<String>,
    #[serde(default)]
    pub upgraded_description_plain: Option<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub upgrades: Vec<WikiCardUpgrade>,
    #[serde(default)]
    pub vars: Vec<WikiCardVar>,
    #[serde(rename = "type", default)]
    pub card_type: Option<String>,
    #[serde(default)]
    pub rarity: Option<String>,
    #[serde(default)]
    pub energy_cost: Option<i64>,
    #[serde(default)]
    pub upgraded_cost: Option<i64>,
}

/// A wiki `monsters.json` row, trimmed to the HP band fields.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct WikiMonster {
    pub class_name: String,
    #[serde(default)]
    pub min_hp: Option<i64>,
    #[serde(default)]
    pub max_hp: Option<i64>,
    #[serde(default)]
    pub min_hp_base: Option<i64>,
    #[serde(default)]
    pub max_hp_base: Option<i64>,
    #[serde(default)]
    pub hp_ascension: Option<i64>,
}

/// A wiki per-relic / per-potion file, trimmed to what
/// `load_owned_localization` reads (relics first, then potions, each sorted
/// by file name).
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct WikiOwned {
    #[serde(default)]
    pub class_name: String,
    #[serde(default)]
    pub description: String,
}

/// A flat JSON object of strings read in DOCUMENT order.
///
/// `serde_json`'s map type sorts its keys, but `sts2_hint.canonical_maps`
/// builds its title maps by iterating `powers.json` in file order with
/// first-wins / last-wins collision rules, so the order has to survive.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OrderedPairs(pub Vec<(String, String)>);

impl<'de> Deserialize<'de> for OrderedPairs {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PairsVisitor;

        impl<'de> Visitor<'de> for PairsVisitor {
            type Value = OrderedPairs;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a flat JSON object whose values are strings")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some((key, value)) = map.next_entry::<String, String>()? {
                    out.push((key, value));
                }
                Ok(OrderedPairs(out))
            }
        }

        deserializer.deserialize_map(PairsVisitor)
    }
}

/// Every bundled input, parsed.
#[derive(Clone, Debug, Default)]
pub struct Data {
    /// First line of `localization_eng/BUILD.txt` — the vendored build.
    pub loc_build: String,
    /// `localization_eng/cards.json`.
    pub loc_cards: LocMap,
    /// `localization_eng/card_keywords.json`.
    pub loc_card_keywords: LocMap,
    /// `localization_eng/afflictions.json`.
    pub loc_afflictions: LocMap,
    /// `localization_eng/rest_site_ui.json`.
    pub loc_rest_site_ui: LocMap,
    /// `localization_eng/relics.json`.
    pub loc_relics: LocMap,
    /// `localization_eng/potions.json`.
    pub loc_potions: LocMap,
    /// `localization_eng/characters.json`.
    pub loc_characters: LocMap,
    /// `localization_eng/encounters.json` (encounter titles, e.g. a boss's
    /// name for the seed checker).
    pub loc_encounters: LocMap,
    /// `localization_eng/events.json`.
    pub loc_events: LocMap,
    /// `localization_eng/monsters.json`.
    pub loc_monsters: LocMap,
    /// `localization_eng/orbs.json`.
    pub loc_orbs: LocMap,
    /// `localization_eng/powers.json`, in file order (see [`OrderedPairs`]).
    pub loc_powers: OrderedPairs,
    /// Trimmed wiki `cards.json`, in manifest order.
    pub wiki_cards: Vec<WikiCard>,
    /// Trimmed wiki `monsters.json`, in manifest order.
    pub wiki_monsters: Vec<WikiMonster>,
    /// Trimmed wiki `relics/*.json` + `potions/*.json`.
    pub wiki_owned: Vec<WikiOwned>,
    /// Sorted relative paths of every file in the art store.
    pub asset_manifest: Vec<String>,
    /// `creature_scales.json`, keyed by creature-visual slug.
    pub creature_scales: BTreeMap<String, CreatureScale>,
    /// `card_db.json` — the deck analyzer's mechanical card stats, keyed by
    /// loc-key (`workbench/deck_analysis.py`).
    pub deck_card_db: BTreeMap<String, serde_json::Value>,
    /// `card_tags.json` — the deck analyzer's curated role / velocity tags
    /// (its `_meta` entry dropped, as the Python loader does). A non-default
    /// build's is the curated table with that build's overlay
    /// (`data/game/<ver>/card_tags.json`, GH #877) merged over it — the
    /// twin of `deck_analysis.card_tags_for`.
    pub deck_card_tags: BTreeMap<String, serde_json::Value>,
}

/// A game-derived bundle file of build `$version`, staged by `build.rs`
/// into `$OUT_DIR/game/[<ver>/]` (a copy of `data/game/[<ver>/]<name>` when
/// present, a stub otherwise). A non-default build's file staged EMPTY is
/// byte-identical to the default build's (`build.rs` dedups it so a shared
/// table is embedded once), so it reads as the default build's text.
macro_rules! bundled {
    ($version:expr, $name:literal) => {{
        let base: &'static str = include_str!(concat!(env!("OUT_DIR"), "/game/", $name));
        let own: &'static str = match $version {
            GameVersion::V0_107_1 => base,
            GameVersion::V0_111_0 => {
                include_str!(concat!(env!("OUT_DIR"), "/game/v0.111.0/", $name))
            }
            GameVersion::V0_113_0 => {
                include_str!(concat!(env!("OUT_DIR"), "/game/v0.113.0/", $name))
            }
        };
        if own.is_empty() {
            base
        } else {
            own
        }
    }};
}

/// A curated (non-game) file committed under `data/`.
macro_rules! curated {
    ($name:literal) => {
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/", $name))
    };
}

fn parse<T: for<'de> Deserialize<'de>>(name: &str, text: &str) -> T {
    serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("sts2-workbench bundle `data/{name}` is malformed: {e}"))
}

impl Data {
    /// Parse the default build's committed bundle. Prefer [`data`], which
    /// does this once.
    #[must_use]
    pub fn load_bundled() -> Self {
        Self::load_bundled_for(GameVersion::default())
    }

    /// Parse build `version`'s committed bundle. Prefer [`data_for`], which
    /// does this once per build.
    #[must_use]
    pub fn load_bundled_for(version: GameVersion) -> Self {
        Self {
            loc_build: bundled!(version, "localization_eng/BUILD.txt")
                .lines()
                .next()
                .unwrap_or_default()
                .trim()
                .to_string(),
            loc_cards: parse(
                "localization_eng/cards.json",
                bundled!(version, "localization_eng/cards.json"),
            ),
            loc_card_keywords: parse(
                "localization_eng/card_keywords.json",
                bundled!(version, "localization_eng/card_keywords.json"),
            ),
            loc_afflictions: parse(
                "localization_eng/afflictions.json",
                bundled!(version, "localization_eng/afflictions.json"),
            ),
            loc_rest_site_ui: parse(
                "localization_eng/rest_site_ui.json",
                bundled!(version, "localization_eng/rest_site_ui.json"),
            ),
            loc_relics: parse(
                "localization_eng/relics.json",
                bundled!(version, "localization_eng/relics.json"),
            ),
            loc_potions: parse(
                "localization_eng/potions.json",
                bundled!(version, "localization_eng/potions.json"),
            ),
            loc_characters: parse(
                "localization_eng/characters.json",
                bundled!(version, "localization_eng/characters.json"),
            ),
            loc_encounters: parse(
                "localization_eng/encounters.json",
                bundled!(version, "localization_eng/encounters.json"),
            ),
            loc_events: parse(
                "localization_eng/events.json",
                bundled!(version, "localization_eng/events.json"),
            ),
            loc_monsters: parse(
                "localization_eng/monsters.json",
                bundled!(version, "localization_eng/monsters.json"),
            ),
            loc_orbs: parse(
                "localization_eng/orbs.json",
                bundled!(version, "localization_eng/orbs.json"),
            ),
            loc_powers: parse(
                "localization_eng/powers.json",
                bundled!(version, "localization_eng/powers.json"),
            ),
            wiki_cards: parse("wiki_cards.json", bundled!(version, "wiki_cards.json")),
            wiki_monsters: parse(
                "wiki_monsters.json",
                bundled!(version, "wiki_monsters.json"),
            ),
            wiki_owned: parse("wiki_owned.json", bundled!(version, "wiki_owned.json")),
            asset_manifest: parse(
                "asset_manifest.json",
                bundled!(version, "asset_manifest.json"),
            ),
            creature_scales: parse(
                "creature_scales.json",
                bundled!(version, "creature_scales.json"),
            ),
            deck_card_db: parse("card_db.json", bundled!(version, "card_db.json")),
            deck_card_tags: {
                let mut tags: BTreeMap<String, serde_json::Value> =
                    parse("card_tags.json", curated!("card_tags.json"));
                tags.remove("_meta");
                let overlay: &'static str = match version {
                    GameVersion::V0_107_1 => "{}",
                    GameVersion::V0_111_0 => {
                        include_str!(concat!(env!("OUT_DIR"), "/game/v0.111.0/card_tags.json"))
                    }
                    GameVersion::V0_113_0 => {
                        include_str!(concat!(env!("OUT_DIR"), "/game/v0.113.0/card_tags.json"))
                    }
                };
                let overlay: BTreeMap<String, serde_json::Value> =
                    parse("game/<ver>/card_tags.json", overlay);
                for (cid, entry) in overlay {
                    if cid == "_meta" {
                        continue;
                    }
                    if entry.is_null() {
                        tags.remove(&cid);
                    } else {
                        tags.insert(cid, entry);
                    }
                }
                tags
            },
        }
    }
}

/// The default build's parsed bundle, built on first use and shared
/// thereafter. Same as `data_for(GameVersion::default())`.
pub fn data() -> &'static Data {
    data_for(GameVersion::default())
}

/// Build `version`'s parsed bundle, built on first use and shared
/// thereafter.
pub fn data_for(version: GameVersion) -> &'static Data {
    static CELLS: PerVersion<Data> = PerVersion::new();
    CELLS.get_or_init(version, || Data::load_bundled_for(version))
}

/// One lazily built value per [`GameVersion`] — the cache behind every
/// `*_for(version)` accessor, so a new build needs no new statics.
pub(crate) struct PerVersion<T>([OnceLock<T>; GameVersion::ALL.len()]);

impl<T> PerVersion<T> {
    pub(crate) const fn new() -> Self {
        Self([const { OnceLock::new() }; GameVersion::ALL.len()])
    }

    pub(crate) fn get_or_init(&self, version: GameVersion, init: impl FnOnce() -> T) -> &T {
        let i = GameVersion::ALL
            .iter()
            .position(|v| *v == version)
            .expect("every GameVersion is in GameVersion::ALL");
        self.0[i].get_or_init(init)
    }
}

/// The game builds this binary can display: the default build plus every
/// other build whose bundle was complete at compile time (`build.rs`). A build
/// left out would render loc keys, so the version picker must not offer it.
pub fn embedded_versions() -> Vec<GameVersion> {
    let complete: Vec<&str> = env!("WORKBENCH_EMBEDDED_VERSIONS").split(',').collect();
    GameVersion::ALL
        .into_iter()
        .filter(|v| v.is_default() || complete.contains(&v.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `build.rs`'s `VERSIONED` is every non-default `GameVersion`: a build
    /// missing there is never staged, one extra is staged for nothing.
    #[test]
    fn build_script_versions_match_game_version_all() {
        let staged: Vec<&str> = env!("WORKBENCH_VERSIONED").split(',').collect();
        let expected: Vec<&str> = GameVersion::ALL
            .into_iter()
            .filter(|v| !v.is_default())
            .map(GameVersion::as_str)
            .collect();
        assert_eq!(staged, expected);
    }

    #[test]
    fn embedded_versions_always_offer_the_default() {
        let v = embedded_versions();
        assert!(v.contains(&GameVersion::default()), "{v:?}");
        #[cfg(workbench_game_data_v0_111_0)]
        assert!(v.contains(&GameVersion::V0_111_0), "{v:?}");
        #[cfg(not(workbench_game_data_v0_111_0))]
        assert!(!v.contains(&GameVersion::V0_111_0), "{v:?}");
    }

    #[cfg(workbench_game_data)]
    #[test]
    fn bundle_parses_and_is_the_pinned_build() {
        let d = data();
        assert_eq!(d.loc_build, ENGINE_WIKI_VERSION);
        assert!(!d.loc_cards.is_empty());
        assert!(!d.wiki_cards.is_empty());
        assert!(!d.wiki_monsters.is_empty());
        assert!(!d.wiki_owned.is_empty());
        assert!(!d.asset_manifest.is_empty());
        assert!(!d.creature_scales.is_empty());
        assert!(!d.loc_powers.0.is_empty());
        // The art manifest is what the resolver's existence checks run
        // against; it is sorted so lookups by prefix are deterministic.
        assert!(d.asset_manifest.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn data_is_the_default_builds_bundle() {
        assert!(std::ptr::eq(data(), data_for(GameVersion::V0_107_1)));
        assert!(!std::ptr::eq(data(), data_for(GameVersion::V0_111_0)));
        assert_eq!(
            engine_wiki_version(GameVersion::V0_107_1),
            ENGINE_WIKI_VERSION
        );
        assert_eq!(engine_wiki_version(GameVersion::V0_111_0), "v0.111.0");
        for v in GameVersion::ALL {
            assert_eq!(engine_wiki_version(v), v.as_str());
        }
    }

    #[cfg(workbench_game_data_v0_111_0)]
    #[test]
    fn v0_111_bundle_parses_and_knows_its_new_content() {
        let d = data_for(GameVersion::V0_111_0);
        assert_eq!(d.loc_build, engine_wiki_version(GameVersion::V0_111_0));
        // Cards, a relic and a potion new in v0.111.0.
        assert_eq!(d.loc_cards["CACOPHONY.title"], "Cacophony");
        assert!(d.wiki_cards.iter().any(|c| c.loc_key == "CACOPHONY"));
        assert!(d.wiki_cards.iter().any(|c| c.loc_key == "BLADE_SYMPHONY"));
        assert!(d.loc_relics.contains_key("DOWSING_ROD.title"));
        assert!(d.wiki_owned.iter().any(|o| o.class_name == "DowsingRod"));
        assert!(d.loc_potions.contains_key("AMBERGRIS.title"));
        // The default build knows none of it.
        let base = data();
        assert!(!base.loc_cards.contains_key("CACOPHONY.title"));
        assert!(!base.wiki_cards.iter().any(|c| c.loc_key == "CACOPHONY"));
        assert!(!base.wiki_owned.iter().any(|o| o.class_name == "DowsingRod"));
        // Content the engine still implements keeps its name (the vendor
        // merge retains keys only the older build had).
        assert_eq!(d.loc_cards["GRAPPLE.title"], "Grapple");
        // The art manifest is the shared tree plus this build's overlay
        // (`v0.111.0/…`); the default build's lists no overlay at all.
        let shared: Vec<&String> = d
            .asset_manifest
            .iter()
            .filter(|r| !r.starts_with("v0.111.0/"))
            .collect();
        assert_eq!(shared, base.asset_manifest.iter().collect::<Vec<_>>());
        assert!(d
            .asset_manifest
            .iter()
            .any(|r| r == "v0.111.0/card_atlas/silent/blade_symphony.png"));
        assert!(!base.asset_manifest.iter().any(|r| r.starts_with("v0.")));
        // The deck analyzer's tables are the build's own (GH #877): the new
        // cards are known and tagged, the removed one is gone.
        assert!(d.deck_card_db.contains_key("CACOPHONY"));
        assert!(!base.deck_card_db.contains_key("CACOPHONY"));
        assert!(!d.deck_card_db.contains_key("SCARE"));
        let db_keys: Vec<&String> = d.deck_card_db.keys().collect();
        let tag_keys: Vec<&String> = d.deck_card_tags.keys().collect();
        assert_eq!(db_keys, tag_keys);
        assert!(!d.creature_scales.is_empty());
        assert!(!d.loc_afflictions.is_empty());
    }

    #[test]
    fn ordered_pairs_keep_document_order() {
        let pairs: OrderedPairs = serde_json::from_str(r#"{"Z.title":"z","A.title":"a"}"#).unwrap();
        assert_eq!(
            pairs.0,
            vec![
                ("Z.title".to_string(), "z".to_string()),
                ("A.title".to_string(), "a".to_string())
            ]
        );
    }

    #[test]
    fn wiki_card_fixture_can_be_sparse() {
        let cards: Vec<WikiCard> =
            serde_json::from_value(serde_json::json!([{"loc_key": "BASH", "title": "Bash"}]))
                .unwrap();
        assert_eq!(cards[0].loc_key, "BASH");
        assert_eq!(cards[0].title.as_deref(), Some("Bash"));
        assert!(cards[0].vars.is_empty());
    }
}
