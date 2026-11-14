//! Resolve STS2 game keys to image paths under the extracted art root.
//!
//! Port of `sts2-pycli/python/workbench/assets.py` in MANIFEST mode only:
//! existence is a set-membership check against the list of relative paths
//! that exist in the art store (the bucket), so the art tree need not be
//! present — which is the only mode that can exist in a browser. Mirrors
//! the sts2-wiki naming convention (`card_atlas/<character>/<slug>.png`,
//! `monsters/<slug>.png`, `intent_atlas/...`, `power_atlas/...`,
//! `sprite_fonts/...`). Every method returns a path relative to the art
//! root, or `None` when the art is absent — exactly what the Python
//! resolver returns, so the enrichment layer's output is byte-identical.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use serde::Serialize;

use crate::data::{self, CreatureScale};
use crate::localization::py_capitalize;

/// Event titles whose slug differs from the extracted illustration filename.
pub const EVENT_ALIASES: &[(&str, &str)] = &[
    ("reflections_snoitcelfer", "reflections"),
    ("welcome_to_wongo_s", "welcome_to_wongos"),
];

/// Monster display names whose slug differs from the extracted sprite
/// filename. Keyed by the slugified *display* name (what the engine sends as
/// `name`), valued by the extracted art slug. Mirrors `MONSTER_IMAGE_ALIASES`
/// in the web frontend (`sts2-wiki/site/src/utils.ts`); keep the two in
/// sync. `monster_rig_coverage` tests fail if a modeled monster stops
/// resolving to art.
pub const MONSTER_ALIASES: &[(&str, &str)] = &[
    ("flyconid", "flying_mushrooms"),
    ("cultist", "cultists"),
    ("assassin_raider", "assassin_ruby_raider"),
    ("axe_raider", "axe_ruby_raider"),
    ("brute_raider", "brute_ruby_raider"),
    ("crossbow_raider", "crossbow_ruby_raider"),
    ("tracker_raider", "tracker_ruby_raider"),
    // Bowlbug variants share the base sprite.
    ("bowlbug_egg", "bowlbug"),
    ("bowlbug_nectar", "bowlbug"),
    ("bowlbug_rock", "bowlbug"),
    ("bowlbug_silk", "bowlbug"),
    // Crusher (left arm) and Rocket (right arm) are the two targetable parts
    // of the Kaiser Crab boss; both share its sprite/rig.
    ("crusher", "kaiser_crab"),
    ("rocket", "kaiser_crab"),
    // Hatchling is Tough Egg's post-hatch display title — same creature.
    ("hatchling", "tough_egg"),
    // Monsters whose sprite is filed under a different name.
    ("torch_head_amalgam", "amalgam"),
    ("globe_head", "orb_head"),
    ("living_fog", "living_smog"),
    ("skulking_colony", "living_shield"),
    ("ovicopter", "egg_layer"),
    // Cultist variants share the base "cultists" sprite/rig.
    ("calcified_cultist", "cultists"),
    ("damp_cultist", "cultists"),
    // Mysterious Knight is a code subclass of Flail Knight (no own art).
    ("mysterious_knight", "flail_knight"),
    // The Merchant??? (FakeMerchantMonster) — rig filed under the class slug.
    ("merchant", "fake_merchant_monster"),
    // Aeonglass has no rig in v0.107.1 — the game renders a static
    // hourglass placeholder sprite.
    ("aeonglass", "hourglass_placeholder"),
    // Battle Friend / dummy variants share the battleworn dummy sprite/rig.
    ("battle_friend_v1_0", "battleworn_dummy"),
    ("battle_friend_v2_0", "battleworn_dummy"),
    ("battle_friend_v3_0", "battleworn_dummy"),
    ("big_dummy", "battleworn_dummy"),
];

/// Monsters that are TWO (or more) targetable combatants sharing ONE big
/// Spine rig. Keyed by the slugified display name, valued by
/// `(group, part)`: `group` ties the shared rig together; `part` selects
/// the animation-track family that combatant drives.
pub const MONSTER_RIG_GROUPS: &[(&str, (&str, &str))] = &[
    ("crusher", ("kaiser_crab", "left")), // Kaiser Crab left arm
    ("rocket", ("kaiser_crab", "right")), // Kaiser Crab right arm
];

/// Monsters whose rig must be picked by their canonical KIND, not their
/// display name — the three Decimillipede segments all localize to the
/// single name "Decimillipede" but each has its own Spine rig. Keyed by
/// `MonsterKind` (PascalCase).
pub const MONSTER_KIND_RIGS: &[(&str, &str)] = &[
    ("DecimillipedeSegmentFront", "decimillipede_segment_front"),
    ("DecimillipedeSegmentMiddle", "decimillipede_segment_middle"),
    ("DecimillipedeSegmentBack", "decimillipede_segment_back"),
];

/// Multi-part creatures keyed by KIND (when the display name can't
/// distinguish the parts). `(group, part)`.
pub const MONSTER_KIND_GROUPS: &[(&str, (&str, &str))] = &[
    ("DecimillipedeSegmentFront", ("decimillipede", "front")),
    ("DecimillipedeSegmentMiddle", ("decimillipede", "middle")),
    ("DecimillipedeSegmentBack", ("decimillipede", "back")),
];

/// Relic display names whose slug differs from the extracted atlas
/// filename. Keyed by the slugified DISPLAY name, valued by the art slug.
pub const RELIC_ALIASES: &[(&str, &str)] = &[
    // FakeMerchantsRug's title is "The Merchant's Rug???"; the sprite keeps
    // the class name's "fake_" prefix like every other Merchant??? counterfeit.
    ("merchants_rug", "fake_merchants_rug"),
];

/// Potion display names whose slug differs from the extracted atlas filename.
pub const POTION_ALIASES: &[(&str, &str)] = &[
    // Clarity's title is "Clarity Extract"; the sprite is filed under the id.
    ("clarity_extract", "clarity"),
];

fn lookup<'a>(table: &'a [(&'a str, &'a str)], key: &str) -> Option<&'a str> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

fn lookup_pair<'a>(
    table: &'a [(&'a str, (&'a str, &'a str))],
    key: &str,
) -> Option<(&'a str, &'a str)> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// The wire's monster kind as the `MonsterKind` the tables above are keyed
/// by. The engine emits PascalCase (`"DecimillipedeSegmentFront"`);
/// SpireMentor emits the game's SCREAMING_SNAKE id
/// (`"DECIMILLIPEDE_SEGMENT_FRONT"`). A kind carrying any lowercase letter
/// is already a `MonsterKind` and is returned untouched; None for an
/// empty/absent kind.
#[must_use]
pub fn normalize_kind(kind: Option<&str>) -> Option<String> {
    let kind = kind.filter(|k| !k.is_empty())?;
    if kind.chars().any(char::is_lowercase) {
        return Some(kind.to_string());
    }
    Some(
        kind.split('_')
            .filter(|part| !part.is_empty())
            .map(py_capitalize)
            .collect(),
    )
}

/// `[^a-z0-9]+` -> `_`, then strip leading/trailing `_`.
fn collapse_non_alnum(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_run = false;
    for c in s.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            out.push(c);
            in_run = false;
        } else if !in_run {
            out.push('_');
            in_run = true;
        }
    }
    out.trim_matches('_').to_string()
}

/// Apostrophes are dropped (not converted to `_`) so "Pael's Legion" ->
/// `paels_legion`.
fn drop_apostrophes(s: &str) -> String {
    s.chars()
        .filter(|c| *c != '\'' && *c != '\u{2019}')
        .collect()
}

/// The raw (article-keeping) slug: lower-cased, apostrophes dropped,
/// non-alphanumerics collapsed to `_`.
fn raw_slug(name: &str) -> String {
    collapse_non_alnum(&drop_apostrophes(&name.trim().to_lowercase()))
}

/// Lowercase, drop a leading "the", collapse non-alphanumerics to `_`.
#[must_use]
pub fn slugify(name: &str) -> String {
    let s = raw_slug(name);
    match s.strip_prefix("the_") {
        Some(rest) => rest.to_string(),
        None => s,
    }
}

/// `(stripped, raw)` slugs for `name` — with and without a leading "the".
/// `monsters/` is generated with the strip (`monsters/kin.png`) while
/// `relic_atlas/` and `power_atlas/` keep it (`the_boot.png`).
#[must_use]
pub fn slug_forms(name: &str) -> (String, String) {
    (slugify(name), raw_slug(name))
}

/// `{"group", "part"}` for a multi-part monster.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MonsterRigGroup {
    pub group: String,
    pub part: String,
}

/// Maps game entities to existing image files in the art store.
#[derive(Clone, Debug, Default)]
pub struct AssetResolver {
    manifest: BTreeSet<String>,
    /// Sorted top-level `card_atlas/<folder>/` prefixes in the manifest.
    card_folders: Vec<String>,
    /// Per-creature authored scene scale + Bounds box, keyed by
    /// creature_visuals slug. Empty when absent — combatants then size by
    /// their fallback height.
    scales: BTreeMap<String, CreatureScale>,
    /// A game build's art overlay folder (`v0.111.0`), tried before the
    /// shared tree: the art that build added or redrew, under the shared
    /// layout (GH #801). `None` for the default build.
    overlay: Option<String>,
}

impl AssetResolver {
    /// A resolver over `manifest` (relative paths that exist in the art
    /// store) and `creature_scales` (the parsed `creature_scales.json`).
    pub fn new<I, S>(manifest: I, creature_scales: BTreeMap<String, CreatureScale>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let manifest: BTreeSet<String> = manifest.into_iter().map(Into::into).collect();
        let card_folders: BTreeSet<String> = manifest
            .iter()
            .filter_map(|rel| {
                let parts: Vec<&str> = rel.split('/').collect();
                (parts.len() >= 3 && parts[0] == "card_atlas").then(|| parts[1].to_string())
            })
            .collect();
        Self {
            manifest,
            card_folders: card_folders.into_iter().collect(),
            scales: creature_scales,
            overlay: None,
        }
    }

    /// This resolver, trying `<overlay>/<rel>` before `<rel>` for every
    /// lookup (the twin of `workbench.assets.AssetResolver(overlay=…)`).
    #[must_use]
    pub fn with_overlay(mut self, overlay: impl Into<String>) -> Self {
        self.overlay = Some(overlay.into());
        self
    }

    /// A resolver over `manifest` alone (no `creature_scales.json`).
    pub fn from_manifest<I, S>(manifest: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::new(manifest, BTreeMap::new())
    }

    /// The resolver over the default build's embedded art manifest +
    /// creature scales, built once.
    pub fn embedded() -> &'static AssetResolver {
        Self::embedded_for(sts2_core::GameVersion::default())
    }

    /// The resolver over build `version`'s embedded art manifest + creature
    /// scales, built once per build. A later build also reads its own art
    /// overlay (`<version>/…`, see [`AssetResolver::with_overlay`]).
    pub fn embedded_for(version: sts2_core::GameVersion) -> &'static AssetResolver {
        use sts2_core::GameVersion;
        static V0_107_1: OnceLock<AssetResolver> = OnceLock::new();
        static V0_111_0: OnceLock<AssetResolver> = OnceLock::new();
        static V0_113_0: OnceLock<AssetResolver> = OnceLock::new();
        let cell = match version {
            GameVersion::V0_107_1 => &V0_107_1,
            GameVersion::V0_111_0 => &V0_111_0,
            GameVersion::V0_113_0 => &V0_113_0,
        };
        cell.get_or_init(|| {
            let d = data::data_for(version);
            let r = AssetResolver::new(d.asset_manifest.iter().cloned(), d.creature_scales.clone());
            if version == GameVersion::default() {
                r
            } else {
                r.with_overlay(version.as_str())
            }
        })
    }

    fn rel_if_exists(&self, rel: String) -> Option<String> {
        if let Some(overlay) = &self.overlay {
            let own = format!("{overlay}/{rel}");
            if self.manifest.contains(&own) {
                return Some(own);
            }
        }
        self.manifest.contains(&rel).then_some(rel)
    }

    /// `{"scale", "bounds_w", "bounds_h"}` for a character rig, or None.
    #[must_use]
    pub fn character_rig_scale(&self, character: &str) -> Option<&CreatureScale> {
        self.scales.get(&slugify(character))
    }

    /// `{"scale", "bounds_w", "bounds_h"}` for a monster rig, or None.
    ///
    /// Same kind-first slug resolution as [`Self::monster_rig`], with one
    /// exception: the members of a SHARED-SKELETON group (the Kaiser Crab's
    /// Crusher and Rocket) are not drawing their own rig, so they take the
    /// shared rig's `scale` while keeping their own Bounds box — `bounds_w`
    /// is exactly what `NCombatRoom.PositionEnemies` spaces the row by.
    #[must_use]
    pub fn monster_rig_scale(&self, name: &str, kind: Option<&str>) -> Option<CreatureScale> {
        let own = Self::rig_slugs(name, kind)
            .into_iter()
            .find_map(|slug| self.scales.get(&slug));
        let group = self.monster_rig_group(name, kind);
        let rig = self.monster_rig(name, kind);
        if let (Some(group), Some(rig)) = (group, rig) {
            let rig_slug = rig.rsplit('/').next().unwrap_or(&rig);
            // One skeleton for several creatures (the crab's arms), as
            // opposed to a group whose members each have their own rig.
            let shared = rig_slug == group.group;
            if shared {
                if let Some(rig_scale) = self.scales.get(rig_slug) {
                    let mut sized = own.cloned().unwrap_or_default();
                    sized.insert(
                        "scale".to_string(),
                        rig_scale.get("scale").copied().flatten(),
                    );
                    return Some(sized);
                }
            }
        }
        own.cloned()
    }

    /// All top-level atlas folders present under `card_atlas/`, sorted.
    #[must_use]
    pub fn card_folders(&self) -> &[String] {
        &self.card_folders
    }

    /// Resolve a card to its atlas image path.
    ///
    /// Searches, in order: the card's own character folder
    /// (`card_character`), the run character's folder, the shared
    /// status/curse/colorless/token/event/quest folders, and finally every
    /// remaining character folder. Within each folder the id slug is tried
    /// before the name slug, and a `beta/` subfolder after the folder
    /// itself.
    #[must_use]
    pub fn card(
        &self,
        character: &str,
        card_id: &str,
        name: &str,
        card_character: Option<&str>,
    ) -> Option<String> {
        let id_slug = if card_id.is_empty() {
            String::new()
        } else {
            card_id.rsplit('.').next().unwrap_or(card_id).to_lowercase()
        };
        let slugs: Vec<String> = [id_slug, slugify(name)]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();
        if slugs.is_empty() {
            return None;
        }

        let mut folders: Vec<String> = Vec::new();
        for c in [card_character, Some(character)] {
            let s = c.map(slugify).unwrap_or_default();
            if !s.is_empty() && !folders.contains(&s) {
                folders.push(s);
            }
        }
        for shared in ["status", "curse", "colorless", "token", "event", "quest"] {
            if !folders.iter().any(|f| f == shared) {
                folders.push(shared.to_string());
            }
        }

        let find = |search: &[String]| -> Option<String> {
            for slug in &slugs {
                for folder in search {
                    for sub in ["", "beta/"] {
                        if let Some(hit) =
                            self.rel_if_exists(format!("card_atlas/{folder}/{sub}{slug}.png"))
                        {
                            return Some(hit);
                        }
                    }
                }
            }
            None
        };

        if let Some(hit) = find(&folders) {
            return Some(hit);
        }
        let rest: Vec<String> = self
            .card_folders
            .iter()
            .filter(|f| !folders.contains(f))
            .cloned()
            .collect();
        find(&rest)
    }

    /// A character's portrait icon path.
    #[must_use]
    pub fn portrait(&self, character: &str) -> Option<String> {
        self.rel_if_exists(format!(
            "characters/character_icon_{}.png",
            slugify(character)
        ))
    }

    /// A map node's room-type icon (`run_history/<slug>.png`). `Boss` nodes
    /// need the `UPPER_SNAKE` encounter id; the exact slug is tried first,
    /// then the `_boss`-suffixed form.
    #[must_use]
    pub fn room_icon(&self, node_type: Option<&str>, boss_id: Option<&str>) -> Option<String> {
        let node_type = node_type.filter(|t| !t.is_empty())?;
        if node_type == "Boss" {
            let boss_id = boss_id.filter(|b| !b.is_empty())?;
            let slug = boss_id.to_lowercase();
            for cand in [slug.clone(), format!("{slug}_boss")] {
                if let Some(rel) = self.rel_if_exists(format!("run_history/{cand}.png")) {
                    return Some(rel);
                }
            }
            return None;
        }
        let slug = match node_type {
            "Monster" => "monster",
            "Elite" | "MegaElite" => "elite",
            "Shop" => "shop",
            "RestSite" | "Rest" | "Campfire" => "rest_site",
            "Treasure" => "treasure",
            "Event" => "event",
            "Ancient" => "ancient",
            _ => return None,
        };
        self.rel_if_exists(format!("run_history/{slug}.png"))
    }

    /// A character's full-body combat sprite (baked Spine idle cutout).
    #[must_use]
    pub fn character_sprite(&self, character: &str) -> Option<String> {
        self.rel_if_exists(format!("character_sprites/{}.png", slugify(character)))
    }

    /// The base path of a character's live Spine rig (no extension):
    /// `character_rigs/<slug>/<slug>` when the `.skel` exists.
    #[must_use]
    pub fn character_rig(&self, character: &str) -> Option<String> {
        let slug = slugify(character);
        self.rel_if_exists(format!("character_rigs/{slug}/{slug}.skel"))
            .map(|skel| skel.trim_end_matches(".skel").to_string())
    }

    /// A monster name's sprite image path.
    #[must_use]
    pub fn monster(&self, name: &str) -> Option<String> {
        let (stripped, raw) = slug_forms(name);
        let candidates = [
            Some(stripped.as_str()),
            Some(raw.as_str()),
            lookup(MONSTER_ALIASES, &stripped),
            lookup(MONSTER_ALIASES, &raw),
        ];
        for slug in candidates.into_iter().flatten() {
            if slug.is_empty() {
                continue;
            }
            if let Some(hit) = self.rel_if_exists(format!("monsters/{slug}.png")) {
                return Some(hit);
            }
        }
        None
    }

    /// Candidate rig slugs for a monster, kind-specific ones first.
    fn rig_slugs(name: &str, kind: Option<&str>) -> Vec<String> {
        let mut slugs: Vec<String> = Vec::new();
        if let Some(norm) = normalize_kind(kind) {
            if let Some(rig) = lookup(MONSTER_KIND_RIGS, &norm) {
                slugs.push(rig.to_string());
            }
        }
        let (stripped, raw) = slug_forms(name);
        let alias_stripped = lookup(MONSTER_ALIASES, &stripped).map(str::to_string);
        let alias_raw = lookup(MONSTER_ALIASES, &raw).map(str::to_string);
        slugs.push(stripped);
        slugs.push(raw);
        slugs.extend(alias_stripped);
        slugs.extend(alias_raw);
        slugs.retain(|s| !s.is_empty());
        slugs
    }

    /// The base path of a monster's live Spine rig (no extension):
    /// `monster_rigs/<slug>/<slug>` when the `.skel` exists. `kind` (the
    /// wire `MonsterKind`, either spelling) wins over `name`.
    #[must_use]
    pub fn monster_rig(&self, name: &str, kind: Option<&str>) -> Option<String> {
        Self::rig_slugs(name, kind).into_iter().find_map(|slug| {
            self.rel_if_exists(format!("monster_rigs/{slug}/{slug}.skel"))
                .map(|skel| skel.trim_end_matches(".skel").to_string())
        })
    }

    /// `{"group", "part"}` for a multi-part monster, or None.
    #[must_use]
    pub fn monster_rig_group(&self, name: &str, kind: Option<&str>) -> Option<MonsterRigGroup> {
        if let Some(norm) = normalize_kind(kind) {
            if let Some((group, part)) = lookup_pair(MONSTER_KIND_GROUPS, &norm) {
                return Some(MonsterRigGroup {
                    group: group.to_string(),
                    part: part.to_string(),
                });
            }
        }
        let (stripped, raw) = slug_forms(name);
        for slug in [stripped, raw] {
            if let Some((group, part)) = lookup_pair(MONSTER_RIG_GROUPS, &slug) {
                return Some(MonsterRigGroup {
                    group: group.to_string(),
                    part: part.to_string(),
                });
            }
        }
        None
    }

    /// A character's energy gem icon path.
    #[must_use]
    pub fn energy(&self, character: &str) -> Option<String> {
        self.rel_if_exists(format!(
            "sprite_fonts/{}_energy_icon.png",
            slugify(character)
        ))
    }

    /// An enemy intent type's icon path. Attack intents pick one of five
    /// daggers by *total* telegraphed damage (`<5 -> 1, <10 -> 2, <20 -> 3,
    /// <40 -> 4, else 5`); every other type uses `intent_<type>.png`.
    #[must_use]
    pub fn intent(&self, intent_type: &str, total_damage: Option<i64>) -> Option<String> {
        let slug = slugify(intent_type);
        if slug == "attack" {
            let dmg = total_damage.unwrap_or(0);
            let tier = if dmg < 5 {
                1
            } else if dmg < 10 {
                2
            } else if dmg < 20 {
                3
            } else if dmg < 40 {
                4
            } else {
                5
            };
            return self.rel_if_exists(format!("intent_atlas/attack/intent_attack_{tier}.png"));
        }
        self.rel_if_exists(format!("intent_atlas/intent_{slug}.png"))
    }

    /// A power/buff name's 64x64 tray icon path.
    #[must_use]
    pub fn power(&self, name: &str) -> Option<String> {
        let (stripped, raw) = slug_forms(name);
        for slug in [stripped, raw] {
            if let Some(hit) = self.rel_if_exists(format!("power_atlas/{slug}_power.png")) {
                return Some(hit);
            }
        }
        None
    }

    /// A power name's 256x256 `BigIcon` (what the power VFX draw), ending at
    /// `power_big/missing_power.png` like `PowerModel.ResolvedBigIconPath`;
    /// None only when even the placeholder is absent.
    #[must_use]
    pub fn power_big(&self, name: &str) -> Option<String> {
        let (stripped, raw) = slug_forms(name);
        for slug in [stripped, raw] {
            if let Some(hit) = self.rel_if_exists(format!("power_big/{slug}_power.png")) {
                return Some(hit);
            }
        }
        self.rel_if_exists("power_big/missing_power.png".to_string())
    }

    /// A relic name's atlas image path.
    #[must_use]
    pub fn relic(&self, name: &str) -> Option<String> {
        let (stripped, raw) = slug_forms(name);
        let alias = lookup(RELIC_ALIASES, &stripped).map(str::to_string);
        for slug in [Some(stripped), Some(raw), alias].into_iter().flatten() {
            if slug.is_empty() {
                continue;
            }
            if let Some(hit) = self.rel_if_exists(format!("relic_atlas/{slug}.png")) {
                return Some(hit);
            }
        }
        None
    }

    /// A potion name's atlas image path.
    #[must_use]
    pub fn potion(&self, name: &str) -> Option<String> {
        let (stripped, raw) = slug_forms(name);
        let alias = lookup(POTION_ALIASES, &stripped).map(str::to_string);
        for slug in [Some(stripped), Some(raw), alias].into_iter().flatten() {
            if slug.is_empty() {
                continue;
            }
            if let Some(hit) = self.rel_if_exists(format!("potion_atlas/{slug}.png")) {
                return Some(hit);
            }
        }
        None
    }

    /// A Defect orb's atlas image path (`orb_atlas/<type>.png`, keyed by the
    /// lowercased C# class name minus `Orb`).
    #[must_use]
    pub fn orb(&self, orb_type: &str) -> Option<String> {
        self.rel_if_exists(format!("orb_atlas/{}.png", slugify(orb_type)))
    }

    /// An event's illustration path from its localized title. Apostrophes
    /// are NOT dropped here (unlike [`slugify`]) — the alias table bridges
    /// the few titles that differ.
    #[must_use]
    pub fn event_illustration(&self, event_name: &str) -> Option<String> {
        if event_name.is_empty() {
            return None;
        }
        let raw = collapse_non_alnum(&event_name.trim().to_lowercase());
        let stripped = raw.strip_prefix("the_").unwrap_or(&raw).to_string();
        let candidates = [
            Some(raw.as_str()),
            Some(stripped.as_str()),
            lookup(EVENT_ALIASES, &stripped),
            lookup(EVENT_ALIASES, &raw),
        ];
        for slug in candidates.into_iter().flatten() {
            if slug.is_empty() {
                continue;
            }
            if let Some(hit) = self.rel_if_exists(format!("events/{slug}.png")) {
                return Some(hit);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(workbench_game_data)] // only the game-data-gated tests use them
    use crate::display_maps::{monster_title, MONSTER_TITLE_TO_ID};
    use serde_json::json;

    fn resolver(files: &[&str]) -> AssetResolver {
        AssetResolver::from_manifest(files.iter().copied())
    }

    fn scales(v: serde_json::Value) -> BTreeMap<String, CreatureScale> {
        serde_json::from_value(v).unwrap()
    }

    fn scale(v: serde_json::Value) -> CreatureScale {
        serde_json::from_value(v).unwrap()
    }

    // ── test_assets.py ─────────────────────────────────────────────

    #[test]
    fn test_slugify() {
        assert_eq!(slugify("Twin Strike"), "twin_strike");
        assert_eq!(slugify("Scroll of Biting!"), "scroll_of_biting");
        assert_eq!(slugify("The Necrobinder"), "necrobinder"); // leading 'the' dropped
        assert_eq!(slugify("Pael's Legion"), "paels_legion");
        assert_eq!(slugify("Pael\u{2019}s Legion"), "paels_legion");
        assert_eq!(slugify("  The Boot  "), "boot");
        assert_eq!(slugify("The"), "the");
        assert_eq!(slugify(""), "");
        assert_eq!(
            slug_forms("The Boot"),
            ("boot".to_string(), "the_boot".to_string())
        );
    }

    #[test]
    fn resolver_finds_existing_card() {
        let r = resolver(&["card_atlas/ironclad/bash.png"]);
        assert_eq!(
            r.card("Ironclad", "CARD.BASH", "Bash", None).as_deref(),
            Some("card_atlas/ironclad/bash.png")
        );
    }

    #[test]
    fn resolver_returns_none_when_missing() {
        let r = resolver(&[]);
        assert_eq!(r.card("Ironclad", "CARD.NOPE", "Nope", None), None);
        assert_eq!(r.monster("Ghost"), None);
        assert_eq!(r.card("Ironclad", "", "", None), None);
    }

    #[test]
    fn resolver_finds_cross_character_card() {
        let r = resolver(&["card_atlas/defect/claw.png"]);
        assert_eq!(
            r.card("Ironclad", "CARD.CLAW", "Claw", Some("Defect"))
                .as_deref(),
            Some("card_atlas/defect/claw.png")
        );
    }

    #[test]
    fn cross_character_collision_prefers_card_character() {
        let r = resolver(&[
            "card_atlas/defect/scavenge.png",
            "card_atlas/necrobinder/scavenge.png",
        ]);
        assert_eq!(
            r.card("Necrobinder", "CARD.SCAVENGE", "Scavenge", Some("Defect"))
                .as_deref(),
            Some("card_atlas/defect/scavenge.png")
        );
    }

    #[test]
    fn cross_character_fallback_scans_all_folders() {
        let r = resolver(&["card_atlas/defect/beam_cell.png"]);
        assert_eq!(
            r.card("Ironclad", "CARD.BEAM_CELL", "Beam Cell", None)
                .as_deref(),
            Some("card_atlas/defect/beam_cell.png")
        );
        assert_eq!(r.card_folders(), ["defect".to_string()]);
    }

    #[test]
    fn card_search_order_id_before_name_and_beta_subfolder() {
        // The id slug wins over the name slug across ALL folders, and the
        // beta/ subfolder is tried right after its folder.
        let r = resolver(&[
            "card_atlas/colorless/beta/bash.png",
            "card_atlas/ironclad/smash.png",
        ]);
        assert_eq!(
            r.card("Ironclad", "CARD.BASH", "Smash", None).as_deref(),
            Some("card_atlas/colorless/beta/bash.png")
        );
        let r = resolver(&["card_atlas/ironclad/smash.png"]);
        assert_eq!(
            r.card("Ironclad", "CARD.BASH", "Smash", None).as_deref(),
            Some("card_atlas/ironclad/smash.png")
        );
    }

    #[test]
    fn monster_alias_resolves_flyconid() {
        let r = resolver(&["monsters/flying_mushrooms.png"]);
        assert_eq!(
            r.monster("Flyconid").as_deref(),
            Some("monsters/flying_mushrooms.png")
        );
    }

    #[test]
    fn monster_alias_resolves_ruby_raiders() {
        let files: Vec<String> = ["axe", "tracker", "brute", "crossbow", "assassin"]
            .iter()
            .map(|s| format!("monsters/{s}_ruby_raider.png"))
            .collect();
        let r = AssetResolver::from_manifest(files);
        assert_eq!(
            r.monster("Axe Raider").as_deref(),
            Some("monsters/axe_ruby_raider.png")
        );
        assert_eq!(
            r.monster("Tracker Raider").as_deref(),
            Some("monsters/tracker_ruby_raider.png")
        );
        assert_eq!(
            r.monster("Brute Raider").as_deref(),
            Some("monsters/brute_ruby_raider.png")
        );
        assert_eq!(
            r.monster("Crossbow Raider").as_deref(),
            Some("monsters/crossbow_ruby_raider.png")
        );
        assert_eq!(
            r.monster("Assassin Raider").as_deref(),
            Some("monsters/assassin_ruby_raider.png")
        );
    }

    #[test]
    fn mysterious_knight_uses_flail_knight_art() {
        let r = resolver(&[
            "monsters/flail_knight.png",
            "monster_rigs/flail_knight/flail_knight.skel",
        ]);
        assert_eq!(
            r.monster("Mysterious Knight").as_deref(),
            Some("monsters/flail_knight.png")
        );
        assert_eq!(
            r.monster_rig("Mysterious Knight", Some("MysteriousKnight"))
                .as_deref(),
            Some("monster_rigs/flail_knight/flail_knight")
        );
    }

    #[test]
    fn fake_merchant_uses_fake_merchant_monster_art() {
        let r = resolver(&["monster_rigs/fake_merchant_monster/fake_merchant_monster.skel"]);
        assert_eq!(
            r.monster_rig("The Merchant???", Some("FakeMerchantMonster"))
                .as_deref(),
            Some("monster_rigs/fake_merchant_monster/fake_merchant_monster")
        );
    }

    #[test]
    fn monster_rig_scale_describes_the_rig_that_is_sent() {
        let r = AssetResolver::new(
            ["monster_rigs/kaiser_crab/kaiser_crab.skel"],
            scales(json!({
                "crusher": {"scale": 1.5, "bounds_h": 366.0},
                "rocket": {"scale": -1.5, "bounds_h": 331.0},
                "kaiser_crab": {"scale": 0.5, "bounds_h": null},
                "nibbit": {"scale": 0.28, "bounds_h": 146.0},
            })),
        );
        assert_eq!(
            r.monster_rig("Crusher", None).as_deref(),
            Some("monster_rigs/kaiser_crab/kaiser_crab")
        );
        // The RIG's scale, the ARM's own Bounds box.
        assert_eq!(
            r.monster_rig_scale("Crusher", None),
            Some(scale(json!({"scale": 0.5, "bounds_h": 366.0})))
        );
        assert_eq!(
            r.monster_rig_scale("Rocket", None),
            Some(scale(json!({"scale": 0.5, "bounds_h": 331.0})))
        );
        // A monster with no staged rig still reports its own authored scale.
        assert_eq!(
            r.monster_rig_scale("Nibbit", None),
            Some(scale(json!({"scale": 0.28, "bounds_h": 146.0})))
        );
        assert_eq!(r.monster_rig_scale("Ghost", None), None);
        // Serialises like the Python dict.
        assert_eq!(
            serde_json::to_string(&r.monster_rig_scale("Crusher", None).unwrap()).unwrap(),
            r#"{"bounds_h":366.0,"scale":0.5}"#
        );
    }

    #[test]
    fn monster_rig_scale_keeps_a_variant_own_scale() {
        let r = AssetResolver::new(
            ["monster_rigs/flail_knight/flail_knight.skel"],
            scales(json!({
                "mysterious_knight": {"scale": 0.17, "bounds_h": 437.0},
                "flail_knight": {"scale": 0.14, "bounds_h": 372.0},
            })),
        );
        assert_eq!(
            r.monster_rig("Mysterious Knight", None).as_deref(),
            Some("monster_rigs/flail_knight/flail_knight")
        );
        assert_eq!(
            r.monster_rig_scale("Mysterious Knight", None),
            Some(scale(json!({"scale": 0.17, "bounds_h": 437.0})))
        );
    }

    #[test]
    fn monster_rig_group_kaiser_crab_arms() {
        let r = resolver(&[]);
        let g = |group: &str, part: &str| {
            Some(MonsterRigGroup {
                group: group.into(),
                part: part.into(),
            })
        };
        assert_eq!(
            r.monster_rig_group("Crusher", None),
            g("kaiser_crab", "left")
        );
        assert_eq!(
            r.monster_rig_group("Rocket", None),
            g("kaiser_crab", "right")
        );
        assert_eq!(r.monster_rig_group("Nibbit", None), None);
        assert_eq!(r.monster_rig_group("", None), None);
        // Decimillipede segments group by kind (all share the name "Decimillipede").
        assert_eq!(
            r.monster_rig_group("Decimillipede", Some("DecimillipedeSegmentFront")),
            g("decimillipede", "front")
        );
        assert_eq!(
            r.monster_rig_group("Decimillipede", Some("DecimillipedeSegmentBack")),
            g("decimillipede", "back")
        );
        assert_eq!(r.monster_rig_group("Decimillipede", None), None); // name alone can't tell
        assert_eq!(
            serde_json::to_string(&r.monster_rig_group("Crusher", None).unwrap()).unwrap(),
            r#"{"group":"kaiser_crab","part":"left"}"#
        );
    }

    #[test]
    fn monster_rig_by_kind_decimillipede_segments() {
        let files: Vec<String> = ["front", "middle", "back"]
            .iter()
            .map(|p| {
                format!("monster_rigs/decimillipede_segment_{p}/decimillipede_segment_{p}.skel")
            })
            .collect();
        let r = AssetResolver::from_manifest(files);
        assert_eq!(
            r.monster_rig("Decimillipede", Some("DecimillipedeSegmentFront"))
                .as_deref(),
            Some("monster_rigs/decimillipede_segment_front/decimillipede_segment_front")
        );
        assert_eq!(
            r.monster_rig("Decimillipede", Some("DecimillipedeSegmentBack"))
                .as_deref(),
            Some("monster_rigs/decimillipede_segment_back/decimillipede_segment_back")
        );
        // Name alone (no kind) can't pick a segment.
        assert_eq!(r.monster_rig("Decimillipede", None), None);
        // An unknown kind falls back to name resolution (still None here).
        assert_eq!(r.monster_rig("Decimillipede", Some("SomethingElse")), None);
    }

    #[test]
    fn relic_and_potion() {
        let r = resolver(&["relic_atlas/akabeko.png", "potion_atlas/fire_potion.png"]);
        assert_eq!(
            r.relic("Akabeko").as_deref(),
            Some("relic_atlas/akabeko.png")
        );
        assert_eq!(
            r.potion("Fire Potion").as_deref(),
            Some("potion_atlas/fire_potion.png")
        );
        assert_eq!(r.relic("Nonexistent"), None);
    }

    #[test]
    fn relic_power_potion_keep_a_leading_the() {
        let r = resolver(&[
            "relic_atlas/the_boot.png",
            "relic_atlas/fake_merchants_rug.png",
            "power_atlas/the_bomb_power.png",
            "power_atlas/strength_power.png",
            "potion_atlas/clarity.png",
        ]);
        assert_eq!(
            r.relic("The Boot").as_deref(),
            Some("relic_atlas/the_boot.png")
        );
        assert_eq!(
            r.power("The Bomb").as_deref(),
            Some("power_atlas/the_bomb_power.png")
        );
        // The stripped form still wins where the file has no article.
        assert_eq!(
            r.power("Strength").as_deref(),
            Some("power_atlas/strength_power.png")
        );
        // Alias tables cover the two names that match neither form.
        assert_eq!(
            r.relic("The Merchant's Rug???").as_deref(),
            Some("relic_atlas/fake_merchants_rug.png")
        );
        assert_eq!(
            r.potion("Clarity Extract").as_deref(),
            Some("potion_atlas/clarity.png")
        );
    }

    #[test]
    fn test_orb() {
        let files: Vec<String> = ["lightning", "frost", "dark", "plasma", "glass"]
            .iter()
            .map(|s| format!("orb_atlas/{s}.png"))
            .collect();
        let r = AssetResolver::from_manifest(files);
        assert_eq!(
            r.orb("Lightning").as_deref(),
            Some("orb_atlas/lightning.png")
        );
        assert_eq!(r.orb("Frost").as_deref(), Some("orb_atlas/frost.png"));
        assert_eq!(r.orb("Plasma").as_deref(), Some("orb_atlas/plasma.png"));
        assert_eq!(r.orb("Nonexistent"), None);
    }

    #[test]
    fn test_event_illustration() {
        let r = resolver(&[
            "events/this_or_that.png",
            "events/the_future_of_potions.png", // file keeps leading 'the_'
            "events/welcome_to_wongos.png",     // aliased
        ]);
        assert_eq!(
            r.event_illustration("This or That?").as_deref(),
            Some("events/this_or_that.png")
        );
        assert_eq!(
            r.event_illustration("The Future of Potions?").as_deref(),
            Some("events/the_future_of_potions.png")
        );
        assert_eq!(
            r.event_illustration("Welcome to Wongo's").as_deref(),
            Some("events/welcome_to_wongos.png")
        );
        assert_eq!(r.event_illustration("The Architect"), None);
        assert_eq!(r.event_illustration(""), None);
    }

    #[test]
    fn room_icon_common_types() {
        let files: Vec<String> = [
            "monster",
            "elite",
            "shop",
            "rest_site",
            "treasure",
            "event",
            "ancient",
        ]
        .iter()
        .map(|s| format!("run_history/{s}.png"))
        .collect();
        let r = AssetResolver::from_manifest(files);
        assert_eq!(
            r.room_icon(Some("Monster"), None).as_deref(),
            Some("run_history/monster.png")
        );
        assert_eq!(
            r.room_icon(Some("Elite"), None).as_deref(),
            Some("run_history/elite.png")
        );
        assert_eq!(
            r.room_icon(Some("Shop"), None).as_deref(),
            Some("run_history/shop.png")
        );
        assert_eq!(
            r.room_icon(Some("RestSite"), None).as_deref(),
            Some("run_history/rest_site.png")
        );
        assert_eq!(
            r.room_icon(Some("Treasure"), None).as_deref(),
            Some("run_history/treasure.png")
        );
        assert_eq!(
            r.room_icon(Some("Event"), None).as_deref(),
            Some("run_history/event.png")
        );
        assert_eq!(
            r.room_icon(Some("Ancient"), None).as_deref(),
            Some("run_history/ancient.png")
        );
        // UI aliases reuse the same badge.
        assert_eq!(
            r.room_icon(Some("MegaElite"), None).as_deref(),
            Some("run_history/elite.png")
        );
        assert_eq!(
            r.room_icon(Some("Rest"), None).as_deref(),
            Some("run_history/rest_site.png")
        );
        assert_eq!(
            r.room_icon(Some("Campfire"), None).as_deref(),
            Some("run_history/rest_site.png")
        );
        // Unknown / None / unmapped -> None (frontend keeps its glyph).
        assert_eq!(r.room_icon(Some("Unknown"), None), None);
        assert_eq!(r.room_icon(None, None), None);
        assert_eq!(r.room_icon(Some(""), None), None);
        assert_eq!(r.room_icon(Some("Boss"), None), None); // Boss needs an id
    }

    #[test]
    fn room_icon_boss_by_id() {
        let r = resolver(&[
            "run_history/vantom_boss.png",
            "run_history/the_insatiable_boss.png",
        ]);
        // bare id -> the `_boss`-suffixed candidate matches
        assert_eq!(
            r.room_icon(Some("Boss"), Some("VANTOM")).as_deref(),
            Some("run_history/vantom_boss.png")
        );
        // id already carrying `_boss` -> exact match wins
        assert_eq!(
            r.room_icon(Some("Boss"), Some("THE_INSATIABLE_BOSS"))
                .as_deref(),
            Some("run_history/the_insatiable_boss.png")
        );
        // unknown boss id / no id -> None
        assert_eq!(r.room_icon(Some("Boss"), Some("MYSTERY_BOSS")), None);
        assert_eq!(r.room_icon(Some("Boss"), None), None);
        assert_eq!(r.room_icon(Some("Boss"), Some("")), None);
    }

    #[test]
    fn monster_kind_accepts_both_producers_spellings() {
        let files: Vec<String> = ["front", "middle", "back"]
            .iter()
            .map(|p| {
                format!("monster_rigs/decimillipede_segment_{p}/decimillipede_segment_{p}.skel")
            })
            .collect();
        let r = AssetResolver::from_manifest(files);
        for kind in ["DecimillipedeSegmentFront", "DECIMILLIPEDE_SEGMENT_FRONT"] {
            assert_eq!(
                r.monster_rig("Decimillipede", Some(kind)).as_deref(),
                Some("monster_rigs/decimillipede_segment_front/decimillipede_segment_front")
            );
            assert_eq!(
                r.monster_rig_group("Decimillipede", Some(kind)),
                Some(MonsterRigGroup {
                    group: "decimillipede".into(),
                    part: "front".into()
                })
            );
        }
    }

    #[test]
    fn normalize_kind_leaves_a_pascal_kind_alone() {
        assert_eq!(
            normalize_kind(Some("DecimillipedeSegmentFront")).as_deref(),
            Some("DecimillipedeSegmentFront")
        );
        assert_eq!(
            normalize_kind(Some("TestSubject")).as_deref(),
            Some("TestSubject")
        );
        assert_eq!(
            normalize_kind(Some("DECIMILLIPEDE_SEGMENT_BACK")).as_deref(),
            Some("DecimillipedeSegmentBack")
        );
        assert_eq!(
            normalize_kind(Some("TWIG_SLIME_S")).as_deref(),
            Some("TwigSlimeS")
        );
        assert_eq!(normalize_kind(Some("")), None);
        assert_eq!(normalize_kind(None), None);
    }

    #[test]
    fn power_big_resolves_the_vfx_icon_and_keeps_the_article() {
        let r = resolver(&[
            "power_atlas/strength_power.png",
            "power_big/strength_power.png",
            "power_big/the_bomb_power.png",
        ]);
        assert_eq!(
            r.power_big("Strength").as_deref(),
            Some("power_big/strength_power.png")
        );
        assert_eq!(
            r.power_big("The Bomb").as_deref(),
            Some("power_big/the_bomb_power.png")
        );
        // The two trees stay independent — the tray icon is NOT what this returns.
        assert_eq!(
            r.power("Strength").as_deref(),
            Some("power_atlas/strength_power.png")
        );
    }

    #[test]
    fn power_big_falls_back_to_missing_power_like_the_game_does() {
        let r = resolver(&["power_big/missing_power.png"]);
        assert_eq!(
            r.power_big("Grapple").as_deref(),
            Some("power_big/missing_power.png")
        );
    }

    #[test]
    fn power_big_is_none_when_the_art_tree_predates_the_pass() {
        let r = resolver(&["power_atlas/strength_power.png"]);
        assert_eq!(r.power_big("Strength"), None);
    }

    #[test]
    fn character_lookups() {
        let r = AssetResolver::new(
            [
                "characters/character_icon_ironclad.png",
                "character_sprites/ironclad.png",
                "character_rigs/ironclad/ironclad.skel",
                "sprite_fonts/ironclad_energy_icon.png",
                "intent_atlas/attack/intent_attack_3.png",
                "intent_atlas/intent_defend.png",
            ],
            scales(json!({"ironclad": {"scale": 0.3, "bounds_w": 200.0, "bounds_h": 400.0}})),
        );
        assert_eq!(
            r.portrait("The Ironclad").as_deref(),
            Some("characters/character_icon_ironclad.png")
        );
        assert_eq!(
            r.character_sprite("Ironclad").as_deref(),
            Some("character_sprites/ironclad.png")
        );
        assert_eq!(
            r.character_rig("Ironclad").as_deref(),
            Some("character_rigs/ironclad/ironclad")
        );
        assert_eq!(r.character_rig("Silent"), None);
        assert_eq!(
            r.energy("Ironclad").as_deref(),
            Some("sprite_fonts/ironclad_energy_icon.png")
        );
        assert_eq!(
            r.character_rig_scale("The Ironclad")
                .and_then(|s| s.get("scale").copied().flatten()),
            Some(0.3)
        );
        assert_eq!(r.character_rig_scale("Silent"), None);
        // Intent tiers.
        assert_eq!(
            r.intent("Attack", Some(12)).as_deref(),
            Some("intent_atlas/attack/intent_attack_3.png")
        );
        assert_eq!(r.intent("Attack", Some(4)), None);
        assert_eq!(r.intent("Attack", None), None);
        assert_eq!(
            r.intent("Defend", None).as_deref(),
            Some("intent_atlas/intent_defend.png")
        );
        assert_eq!(r.intent("Unknown", None), None);
    }

    #[test]
    fn intent_attack_tiers_follow_attack_intent_get_texture() {
        let files: Vec<String> = (1..=5)
            .map(|t| format!("intent_atlas/attack/intent_attack_{t}.png"))
            .collect();
        let r = AssetResolver::from_manifest(files);
        for (dmg, tier) in [
            (0, 1),
            (4, 1),
            (5, 2),
            (9, 2),
            (10, 3),
            (19, 3),
            (20, 4),
            (39, 4),
            (40, 5),
            (999, 5),
        ] {
            assert_eq!(
                r.intent("attack", Some(dmg)).unwrap(),
                format!("intent_atlas/attack/intent_attack_{tier}.png"),
                "{dmg}"
            );
        }
    }

    // ── test_monster_rig_coverage.py ───────────────────────────────

    /// The name the WIRE actually carries for a monster (see the Python
    /// test's `_wire_name`): `display_maps.monster_title` over its id.
    #[cfg(workbench_game_data)] // only the game-data-gated tests use it
    fn wire_name(title: &str) -> String {
        let id = MONSTER_TITLE_TO_ID
            .iter()
            .find(|(t, _)| *t == title)
            .unwrap()
            .1;
        monster_title(id)
    }

    /// Monsters with NO extracted art of any kind.
    const NO_ART: &[&str] = &[
        "The Adversary Mk 1",
        "The Adversary Mk 2",
        "The Adversary Mk 3",
    ];

    /// Monsters with a static sprite but no Spine rig (supersets NO_ART).
    const NO_RIG_EXTRA: &[&str] = &["Aeonglass", "Decimillipede"];

    #[cfg(workbench_game_data)] // only the game-data-gated tests use it
    fn no_rig(title: &str) -> bool {
        NO_ART.contains(&title) || NO_RIG_EXTRA.contains(&title)
    }

    #[test]
    fn an_overlay_wins_over_the_shared_tree_and_only_for_its_build() {
        let manifest = [
            "card_atlas/defect/beta/voltaic.png",
            "v0.111.0/card_atlas/defect/voltaic.png",
            "v0.111.0/relic_atlas/dowsing_rod.png",
        ];
        let base = AssetResolver::from_manifest(manifest);
        let beta = base.clone().with_overlay("v0.111.0");
        assert_eq!(
            base.card("Defect", "CARD.VOLTAIC", "Voltaic", Some("Defect"))
                .as_deref(),
            Some("card_atlas/defect/beta/voltaic.png")
        );
        assert_eq!(
            beta.card("Defect", "CARD.VOLTAIC", "Voltaic", Some("Defect"))
                .as_deref(),
            Some("v0.111.0/card_atlas/defect/voltaic.png")
        );
        assert_eq!(base.relic("Dowsing Rod"), None);
        assert_eq!(
            beta.relic("Dowsing Rod").as_deref(),
            Some("v0.111.0/relic_atlas/dowsing_rod.png")
        );
    }

    #[test]
    fn a_rig_only_the_overlay_has_resolves_to_the_overlay() {
        // The Fool and the v0.113.0 monsters ship their rigs in the overlay
        // only. Mutation caught: the rig lookups checking the shared path
        // alone (no rig at all) or returning it bare (a URL that 404s).
        let manifest = [
            "v0.113.0/character_rigs/fool/fool.skel",
            "v0.113.0/monster_rigs/wuppy/wuppy.skel",
            "monster_rigs/nibbit/nibbit.skel",
        ];
        let base = AssetResolver::from_manifest(manifest);
        let new = base.clone().with_overlay("v0.113.0");
        assert_eq!(base.character_rig("Fool"), None);
        assert_eq!(
            new.character_rig("Fool").as_deref(),
            Some("v0.113.0/character_rigs/fool/fool")
        );
        assert_eq!(
            new.monster_rig("Wuppy", None).as_deref(),
            Some("v0.113.0/monster_rigs/wuppy/wuppy")
        );
        assert_eq!(
            new.monster_rig("Nibbit", None).as_deref(),
            Some("monster_rigs/nibbit/nibbit")
        );
    }

    #[cfg(workbench_game_data_v0_113_0)] // the bundled v0.113.0 art manifest
    #[test]
    fn the_v0_113_bundle_draws_the_fool_and_its_new_monsters() {
        use sts2_core::GameVersion;
        let old = AssetResolver::embedded_for(GameVersion::V0_107_1);
        let new = AssetResolver::embedded_for(GameVersion::V0_113_0);
        assert_eq!(old.character_rig("Fool"), None);
        assert_eq!(
            new.character_rig("Fool").as_deref(),
            Some("v0.113.0/character_rigs/fool/fool")
        );
        assert_eq!(
            new.character_sprite("Fool").as_deref(),
            Some("v0.113.0/character_sprites/fool.png")
        );
        assert_eq!(
            new.energy("Fool").as_deref(),
            Some("v0.113.0/sprite_fonts/fool_energy_icon.png")
        );
        assert!(new.character_rig_scale("Fool").is_some());
        // Every Fool card but Strike/Defend ships only as sketch art.
        assert_eq!(
            new.card("Fool", "CARD.ALL_IS_WELL", "All Is Well", Some("Fool"))
                .as_deref(),
            Some("v0.113.0/card_atlas/fool/beta/all_is_well.png")
        );
        for monster in ["Aeonglass", "Wuppy"] {
            assert!(new.monster_rig(monster, None).is_some(), "{monster}");
        }
    }

    #[cfg(workbench_game_data_v0_111_0)] // the bundled v0.111.0 art manifest
    #[test]
    fn the_v0_111_bundle_draws_its_new_and_redrawn_art() {
        use sts2_core::GameVersion;
        let old = AssetResolver::embedded_for(GameVersion::V0_107_1);
        let new = AssetResolver::embedded_for(GameVersion::V0_111_0);
        let blade = |r: &AssetResolver| {
            r.card(
                "Silent",
                "CARD.BLADE_SYMPHONY",
                "Blade Symphony",
                Some("Silent"),
            )
        };
        assert_eq!(blade(old), None);
        assert_eq!(
            blade(new).as_deref(),
            Some("v0.111.0/card_atlas/silent/blade_symphony.png")
        );
        let voltaic =
            |r: &AssetResolver| r.card("Defect", "CARD.VOLTAIC", "Voltaic", Some("Defect"));
        assert_eq!(
            voltaic(old).as_deref(),
            Some("card_atlas/defect/beta/voltaic.png")
        );
        assert_eq!(
            voltaic(new).as_deref(),
            Some("v0.111.0/card_atlas/defect/voltaic.png")
        );
        // Art both builds share still comes from the shared tree.
        assert_eq!(
            new.card("Ironclad", "CARD.BASH", "Bash", Some("Ironclad"))
                .as_deref(),
            Some("card_atlas/ironclad/bash.png")
        );
    }

    #[cfg(workbench_game_data)] // asserts on real game text (see build.rs)
    #[test]
    fn every_enemy_has_an_image() {
        let assets = AssetResolver::embedded();
        let mut blank = Vec::new();
        for (title, _) in MONSTER_TITLE_TO_ID {
            if NO_ART.contains(title) {
                continue;
            }
            let name = wire_name(title);
            if assets.monster(&name).is_none() && assets.monster_rig(&name, None).is_none() {
                blank.push(*title);
            }
        }
        assert!(
            blank.is_empty(),
            "monsters resolving to no sprite AND no rig (add a MONSTER_ALIASES entry, or NO_ART): {blank:?}"
        );
    }

    #[cfg(workbench_game_data)] // asserts on real game text (see build.rs)
    #[test]
    fn every_enemy_has_a_rig() {
        let assets = AssetResolver::embedded();
        let mut rigless = Vec::new();
        for (title, _) in MONSTER_TITLE_TO_ID {
            if no_rig(title) {
                continue;
            }
            if assets.monster_rig(&wire_name(title), None).is_none() {
                rigless.push(*title);
            }
        }
        assert!(
            rigless.is_empty(),
            "monsters with no Spine rig (add a MONSTER_ALIASES entry, or NO_RIG): {rigless:?}"
        );
    }

    #[cfg(workbench_game_data)] // asserts on real game text (see build.rs)
    #[test]
    fn kind_addressed_rigs_resolve() {
        let assets = AssetResolver::embedded();
        for (kind, slug) in MONSTER_KIND_RIGS {
            assert!(
                assets.monster_rig("", Some(kind)).is_some(),
                "MonsterKind {kind:?} resolves to no Spine rig (expected {slug:?})"
            );
        }
    }

    #[test]
    fn allowlists_are_not_stale() {
        let assets = AssetResolver::embedded();
        let gained_art: Vec<&str> = NO_ART
            .iter()
            .copied()
            .filter(|t| assets.monster(t).is_some() || assets.monster_rig(t, None).is_some())
            .collect();
        assert!(
            gained_art.is_empty(),
            "these NO_ART monsters now have art — remove them: {gained_art:?}"
        );
        let gained_rig: Vec<&str> = NO_ART
            .iter()
            .chain(NO_RIG_EXTRA.iter())
            .copied()
            .filter(|t| assets.monster_rig(t, None).is_some())
            .collect();
        assert!(
            gained_rig.is_empty(),
            "these NO_RIG monsters now have a rig — remove them: {gained_rig:?}"
        );
    }

    #[cfg(workbench_game_data)] // asserts on real game text (see build.rs)
    #[test]
    fn embedded_resolver_reads_the_bundle() {
        let assets = AssetResolver::embedded();
        assert!(assets.card_folders().iter().any(|f| f == "ironclad"));
        assert_eq!(
            assets
                .card("Ironclad", "CARD.BASH", "Bash", None)
                .as_deref(),
            Some("card_atlas/ironclad/bash.png")
        );
        assert!(assets.character_rig_scale("Ironclad").is_some());
        assert!(assets.monster_rig_scale("Nibbit", Some("Nibbit")).is_some());
        // The crab's arms take the shared rig's scale.
        let crusher = assets
            .monster_rig_scale("Crusher", Some("Crusher"))
            .unwrap();
        let crab = assets.monster_rig_scale("Kaiser Crab", None).unwrap();
        assert_eq!(crusher.get("scale"), crab.get("scale"));
    }
}
