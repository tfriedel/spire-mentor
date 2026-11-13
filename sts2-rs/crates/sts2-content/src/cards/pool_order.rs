//! The C# card pools in `GenerateAllCards()` order (GH #991), and the combat
//! generation pools a CO-OP fight draws from.
//!
//! C# builds every generation pool the same way: the pool's
//! `GenerateAllCards()` list, through `CardFactory.FilterForCombat` (no
//! Basic / Ancient / Event rarity, `CanBeGeneratedInCombat`) and
//! `FilterForPlayerCount` (`MultiplayerOnly` cards only when
//! `Players.Count > 1`). The single-player pools are hand-validated
//! elsewhere ([`crate::cards::colorless::helpers`]); a co-op fight needs the
//! same lists with the multiplayer-only cards in their places, because
//! `TakeRandom` / `NextItem` read a pool by position. [`combat_pool`] derives
//! both from the C# order, and its tests hold the single-player derivation
//! to the hand-validated pools.

use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::GameVersion;

use super::types::{CardRarity, CardSpec};

#[path = "pool_order_generated.rs"]
mod generated;

/// A C# card pool a combat generation draws from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    Character(CharacterId),
    Colorless,
}

/// `pool`'s `GenerateAllCards()` list in game build `version`.
#[must_use]
pub fn generate_all_cards(pool: Pool, version: GameVersion) -> &'static [CardId] {
    use generated as g;
    let v0111 = super::v0111::is_v0111(version);
    match (pool, v0111) {
        (Pool::Character(CharacterId::Ironclad), false) => g::IRONCLAD_V0107,
        (Pool::Character(CharacterId::Ironclad), true) => g::IRONCLAD_V0111,
        (Pool::Character(CharacterId::Silent), false) => g::SILENT_V0107,
        (Pool::Character(CharacterId::Silent), true) => g::SILENT_V0111,
        (Pool::Character(CharacterId::Defect), false) => g::DEFECT_V0107,
        (Pool::Character(CharacterId::Defect), true) => g::DEFECT_V0111,
        (Pool::Character(CharacterId::Necrobinder), false) => g::NECROBINDER_V0107,
        (Pool::Character(CharacterId::Necrobinder), true) => g::NECROBINDER_V0111,
        (Pool::Character(CharacterId::Regent), false) => g::REGENT_V0107,
        (Pool::Character(CharacterId::Regent), true) => g::REGENT_V0111,
        (Pool::Colorless, false) => g::COLORLESS_V0107,
        (Pool::Colorless, true) if version >= GameVersion::V0_113_0 => &COLORLESS_V0113,
        (Pool::Colorless, true) => g::COLORLESS_V0111,
    }
}

/// `ColorlessCardPool.GenerateAllCards()`, v0.113.0: the v0.111.0 list with
/// MyShinies right after MindBlast (`ColorlessCardPool.cs:62-63`), the only
/// change. The character pools did not change in v0.113.0; the Event and
/// Token pools did, and only [`in_colorless_pool`] reads those.
static COLORLESS_V0113: std::sync::LazyLock<Vec<CardId>> = std::sync::LazyLock::new(|| {
    let mut ids = generated::COLORLESS_V0111.to_vec();
    let at = ids
        .iter()
        .position(|&id| id == CardId::MindBlast)
        .expect("MindBlast is in ColorlessCardPool");
    ids.insert(at + 1, CardId::MyShinies);
    ids
});

/// Cards v0.113.0 adds to a colorless pool: MyShinies (Colorless) and the
/// event cards DingyDagger, MightyShield, MightyWeapon (Event). The Event
/// pool's GodByrd and the Token pool's tarot cards have no `CardId` yet.
const V0113_COLORLESS_POOL_ADDITIONS: [CardId; 4] = [
    CardId::MyShinies,
    CardId::DingyDagger,
    CardId::MightyShield,
    CardId::MightyWeapon,
];

/// Whether `id` is in a C# card pool whose `IsColorless` is true: the
/// Colorless, Event and Token pools (`ColorlessCardPool.cs`,
/// `EventCardPool.cs`, `TokenCardPool.cs`). The fourth such pool,
/// `DeprecatedCardPool`, holds only `DeprecatedCard`, which has no
/// `CardId`. The character, Status, Curse and Quest pools are not
/// colorless.
///
/// One answer serves both builds: v0.111.0 adds Abundance (Event) and
/// TheBall (Colorless), which no v0.107.1 state contains, and no card
/// changed pools between the builds.
#[must_use]
pub fn in_colorless_pool(id: CardId) -> bool {
    use generated as g;
    [
        g::COLORLESS_V0107,
        g::COLORLESS_V0111,
        g::EVENT_V0107,
        g::EVENT_V0111,
        g::TOKEN_V0107,
        g::TOKEN_V0111,
    ]
    .iter()
    .any(|pool| pool.contains(&id))
        || V0113_COLORLESS_POOL_ADDITIONS.contains(&id)
}

/// Whether `id` overrides C# `VisualCardPool` with a pool whose
/// `IsColorless` is false (#1181): an Event card that wears a character's
/// color (`Caltrops.cs:17`, `HelloWorld.cs:12`, …). Generated from the
/// decomp with the pool lists; the same ten cards in both builds.
#[must_use]
pub fn has_character_visual_pool(id: CardId) -> bool {
    use generated as g;
    g::CHARACTER_VISUAL_POOL_V0107.contains(&id) || g::CHARACTER_VISUAL_POOL_V0111.contains(&id)
}

/// `pool`'s combat generation list in game build `version`, for a fight of
/// one player (`multiplayer == false`) or more: `GenerateAllCards()` order
/// through `FilterForCombat` and `FilterForPlayerCount`, minus the cards an
/// unrevealed epoch locks (`GetUnlockedCards`). Built on every call — the
/// cached entry points are [`crate::cards::colorless::helpers`]'.
#[must_use]
pub fn combat_pool(pool: Pool, version: GameVersion, multiplayer: bool) -> Vec<CardSpec> {
    use crate::cards::colorless::helpers::{
        card_can_be_generated_in_combat, card_is_multiplayer_only,
    };
    generate_all_cards(pool, version)
        .iter()
        .filter_map(|&id| super::spec_index::spec_for_card_id(id))
        .map(|s| super::v0111::spec_for(s, version))
        .filter(|s| {
            !matches!(
                s.rarity,
                CardRarity::Basic | CardRarity::Ancient | CardRarity::Event
            ) && card_can_be_generated_in_combat(s.id)
                && super::v0111::can_be_generated_in_combat(s.id, version)
                && (multiplayer || !card_is_multiplayer_only(s.id))
                && !crate::unlocks::is_card_locked(s.id)
        })
        .collect()
}

/// Whether `state`'s fight has more than one player (C# `Players.Count > 1`,
/// `CardFactory.FilterForPlayerCount`): a dead teammate still counts.
#[must_use]
pub fn is_multiplayer(state: &sts2_core::CombatState) -> bool {
    !state.allies.is_empty()
}

/// [`combat_pool`] for a fight with teammates, cached per pool, build and
/// unlock set (a co-op fight generates cards as often as a solo one; the
/// single-player pools are cached the same way).
#[must_use]
pub fn coop_combat_pool(pool: Pool, version: GameVersion) -> &'static [CardSpec] {
    use std::collections::BTreeMap;
    use std::sync::RwLock;

    type Cache = RwLock<Option<(u64, BTreeMap<(u8, bool), &'static [CardSpec]>)>>;
    static CACHE: Cache = RwLock::new(None);

    let generation = crate::unlocks::locked_generation();
    let key = (
        match pool {
            Pool::Character(c) => c as u8,
            Pool::Colorless => u8::MAX,
        },
        super::v0111::is_v0111(version),
    );
    if let Some((gen, map)) = CACHE.read().expect("pool cache poisoned").as_ref() {
        if *gen == generation {
            if let Some(slice) = map.get(&key) {
                return slice;
            }
        }
    }
    let leaked: &'static [CardSpec] =
        Box::leak(combat_pool(pool, version, true).into_boxed_slice());
    let mut guard = CACHE.write().expect("pool cache poisoned");
    match guard.as_mut() {
        Some((gen, map)) if *gen == generation => {
            map.insert(key, leaked);
        }
        _ => *guard = Some((generation, BTreeMap::from([(key, leaked)]))),
    }
    leaked
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHARACTERS: [CharacterId; 5] = [
        CharacterId::Ironclad,
        CharacterId::Silent,
        CharacterId::Defect,
        CharacterId::Necrobinder,
        CharacterId::Regent,
    ];
    const VERSIONS: [GameVersion; 3] = [
        GameVersion::V0_107_1,
        GameVersion::V0_111_0,
        GameVersion::V0_113_0,
    ];

    fn ids(pool: &[CardSpec]) -> Vec<CardId> {
        pool.iter().map(|s| s.id).collect()
    }

    /// The single-player derivation IS the hand-validated pool, card for
    /// card and in order, for every character and build — so the co-op one,
    /// the same derivation with the multiplayer-only cards kept, is C#'s.
    #[test]
    fn the_single_player_derivation_is_the_validated_pool() {
        use crate::cards::colorless::helpers::combat_generatable_pool_in;
        for version in VERSIONS {
            for character in CHARACTERS {
                assert_eq!(
                    ids(&combat_pool(Pool::Character(character), version, false)),
                    ids(combat_generatable_pool_in(character, version)),
                    "{character:?} {version:?}"
                );
            }
        }
    }

    #[test]
    fn the_single_player_colorless_derivation_is_the_validated_pool() {
        for version in VERSIONS {
            let validated =
                crate::cards::colorless::helpers::colorless_combat_generatable_ids_in(version);
            assert_eq!(
                ids(&combat_pool(Pool::Colorless, version, false)),
                validated.as_slice(),
                "{version:?}"
            );
        }
    }

    /// v0.113.0 `ColorlessCardPool.GenerateAllCards()` read off the decomp.
    #[test]
    fn colorless_v0113_matches_the_decomp() {
        let tag = "v0.113.0";
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../../sts2-wiki/decompiled/{tag}/MegaCrit.Sts2.Core.Models.CardPools/ColorlessCardPool.cs"
        ));
        // The public snapshot ships no decomp; nothing to compare there.
        let Ok(src) = std::fs::read_to_string(&path) else {
            return;
        };
        let body = &src[src.find("GenerateAllCards").expect("list")..];
        let body = &body[..body.find("};").expect("end")];
        let names: Vec<String> = body
            .split("ModelDb.Card<")
            .skip(1)
            .map(|c| c[..c.find('>').expect("close")].to_string())
            .collect();
        let ours: Vec<String> = generate_all_cards(Pool::Colorless, GameVersion::V0_113_0)
            .iter()
            .map(|id| {
                let n = format!("{id:?}");
                if names.contains(&n) {
                    n
                } else {
                    n.trim_end_matches("Card").to_string()
                }
            })
            .collect();
        assert_eq!(ours, names);
    }

    /// [`combat_pool`] skips an id without a spec, which would silently
    /// shorten a pool: every card of every C# pool has one.
    #[test]
    fn every_pool_card_has_a_spec() {
        let pools = CHARACTERS
            .map(Pool::Character)
            .into_iter()
            .chain([Pool::Colorless]);
        for pool in pools {
            for version in VERSIONS {
                for &id in generate_all_cards(pool, version) {
                    assert!(
                        crate::cards::spec_index::spec_for_card_id(id).is_some(),
                        "{id:?} ({pool:?} {version:?}) has no spec"
                    );
                }
            }
        }
    }

    /// Co-op adds the multiplayer-only cards in their C# places: the
    /// colorless pool gains Believe in You between Beat Down and Bolas.
    #[test]
    fn a_coop_pool_keeps_the_multiplayer_only_cards_in_place() {
        let coop = ids(&combat_pool(Pool::Colorless, GameVersion::V0_111_0, true));
        let at = coop
            .iter()
            .position(|&c| c == CardId::BelieveInYou)
            .expect("in co-op");
        assert_eq!(coop[at - 1], CardId::BeatDown);
        assert_eq!(coop[at + 1], CardId::Bolas);
        let ironclad = ids(&combat_pool(
            Pool::Character(CharacterId::Ironclad),
            GameVersion::V0_111_0,
            true,
        ));
        assert!(ironclad.contains(&CardId::Blaze));
        assert!(ironclad.contains(&CardId::DemonicShield));
    }
}
