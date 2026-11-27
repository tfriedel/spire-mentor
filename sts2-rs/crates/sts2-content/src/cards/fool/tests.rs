//! Fool card registration: ids, pool order, specs and lookups.
use super::*;
use crate::cards::spec_index::{is_multiplayer_only, spec_for_card_id};
use crate::cards::types::CardRarity;
use sts2_core::GameVersion;

/// Path of a v0.113.0 decomp file, or `None` when the decomp is absent (the
/// public export ships without it).
/// `file` in the v0.113.0 decomp folder whose namespace ends in `.{ns}`
/// (e.g. `ns = "Models.CardPools"`). `None` when the decomp is absent, as in
/// the public snapshot.
fn decomp(ns: &str, file: &str) -> Option<String> {
    decomp_in("v0.113.0", ns, file)
}

/// [`decomp`] in the decomp folder `tag` (e.g. `"v0.114.0"`).
fn decomp_in(tag: &str, ns: &str, file: &str) -> Option<String> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../../sts2-wiki/decompiled/{tag}"));
    let suffix = format!(".{ns}");
    let dir = std::fs::read_dir(root)
        .ok()?
        .flatten()
        .find(|e| e.file_name().to_string_lossy().ends_with(&suffix))?;
    // A Windows checkout (core.autocrlf) has the decomp in CRLF; the
    // parsers below split on "\n".
    std::fs::read_to_string(dir.path().join(file))
        .ok()
        .map(|s| s.replace("\r\n", "\n"))
}

fn names_in(src: &str, marker: &str) -> Vec<String> {
    let body = &src[src.find(marker).expect("list")..];
    let body = &body[..body.find("};").expect("end")];
    body.split("ModelDb.Card<")
        .skip(1)
        .map(|c| c[..c.find('>').expect("close")].to_string())
        .collect()
}

/// The C# class name a `CardId` stands for: `XCard` is the card `X`.
fn class_name(id: CardId) -> String {
    let n = format!("{id:?}");
    if spec_for_card_id(id)
        .is_some_and(|s| id.to_python_name().ends_with("_CARD") && !s.name.ends_with("_CARD"))
    {
        n.trim_end_matches("Card").to_string()
    } else {
        n
    }
}

#[test]
fn fool_pool_matches_the_decomp() {
    let Some(src) = decomp("Models.CardPools", "FoolCardPool.cs") else {
        return;
    };
    let ours: Vec<String> = FOOL_POOL_V0113.iter().map(|&id| class_name(id)).collect();
    assert_eq!(ours, names_in(&src, "GenerateAllCards"));
    let Some(src) = decomp_in("v0.114.0", "Models.CardPools", "FoolCardPool.cs") else {
        return;
    };
    let ours: Vec<String> = FOOL_POOL_V0114.iter().map(|&id| class_name(id)).collect();
    assert_eq!(ours, names_in(&src, "GenerateAllCards"));
}

/// The v0.114.0 Fool ids: Optimism and TwinShot are new, RapidShot is gone.
#[test]
fn v0114_fool_ids_enter_and_rapid_shot_leaves() {
    for id in CardId::V0114_ONLY {
        assert!(is_fool_card(id), "{id:?}");
        assert_eq!(id.introduced_in(), GameVersion::V0_114_0, "{id:?}");
        assert!(
            !CardId::all_for(GameVersion::V0_113_0).contains(&id),
            "{id:?}"
        );
        assert!(
            CardId::all_for(GameVersion::V0_114_0).contains(&id),
            "{id:?}"
        );
        assert!(FOOL_POOL_V0114.contains(&id), "{id:?}");
        assert!(!FOOL_POOL_V0113.contains(&id), "{id:?}");
        assert_eq!(spec_for_card_id(id).map(|s| s.id), Some(id), "{id:?}");
        assert_eq!(
            crate::names::card_id_from_str(id.to_python_name()),
            Some(id)
        );
    }
    assert!(CardId::all_for(GameVersion::V0_113_0).contains(&CardId::RapidShot));
    assert!(!CardId::all_for(GameVersion::V0_114_0).contains(&CardId::RapidShot));
    assert!(CardId::RapidShot.exists_in(GameVersion::V0_113_0));
    assert!(!CardId::RapidShot.exists_in(GameVersion::V0_114_0));
    assert!(!FOOL_POOL_V0114.contains(&CardId::RapidShot));
    use sts2_core::powers::PowerKind;
    assert!(PowerKind::all_for(GameVersion::V0_113_0).contains(&PowerKind::FreeCard));
    assert!(!PowerKind::all_for(GameVersion::V0_114_0).contains(&PowerKind::FreeCard));
}

/// Every new card class in the v0.114.0 decomp (against the v0.113.1 one
/// `V0_113_0` follows) that sits in a pool the Fool owns is a
/// [`CardId::V0114_ONLY`] id.
#[test]
fn v0114_only_cards_are_the_new_fool_classes() {
    let (Some(new), Some(_)) = (
        decomp_in("v0.114.0", "Models.CardPools", "FoolCardPool.cs"),
        decomp_in("v0.113.1", "Models.CardPools", "FoolCardPool.cs"),
    ) else {
        return;
    };
    let old = decomp_in("v0.113.1", "Models.CardPools", "FoolCardPool.cs").unwrap();
    let old = names_in(&old, "GenerateAllCards");
    let added: Vec<String> = names_in(&new, "GenerateAllCards")
        .into_iter()
        .filter(|c| !old.contains(c))
        .collect();
    let ours: Vec<String> = CardId::V0114_ONLY
        .iter()
        .map(|&id| class_name(id))
        .collect();
    assert_eq!(added, ours);
}

#[test]
fn every_fool_card_is_v0113_only_with_a_spec() {
    let mut seen = std::collections::BTreeSet::new();
    for (i, &id) in FOOL_CARDS.iter().enumerate() {
        assert!(seen.insert(id), "{id:?} listed twice");
        assert_eq!(specs::FOOL_SPECS[i]().id, id, "FOOL_SPECS order");
        if CardId::V0114_ONLY.contains(&id) {
            continue; // `v0114_fool_ids_enter_and_rapid_shot_leaves`
        }
        assert!(CardId::V0113_ONLY.contains(&id), "{id:?}");
        assert_eq!(id.introduced_in(), GameVersion::V0_113_0, "{id:?}");
        assert!(
            !CardId::all_for(GameVersion::V0_111_0).contains(&id),
            "{id:?}"
        );
        assert!(
            CardId::all_for(GameVersion::V0_113_0).contains(&id),
            "{id:?}"
        );
        assert_eq!(specs::FOOL_SPECS[i]().id, id, "FOOL_SPECS order");
        assert_eq!(spec_for_card_id(id).map(|s| s.id), Some(id), "{id:?}");
        // The id-string round trips through both name parsers.
        assert_eq!(
            crate::names::card_id_from_str(id.to_python_name()),
            Some(id)
        );
    }
    for id in FOOL_POOL_V0113 {
        assert!(is_fool_card(id));
    }
}

/// Every new card class in the v0.113.0 decomp is a Fool card, an already
/// registered v0.113.0 card, Dread's Amnesia token (registered as
/// `CardId::Amnesia`, GH #1346, and not a Fool card), or the one card still
/// out of scope (GodByrd, an event card).
#[test]
fn fool_cards_are_every_new_fool_class() {
    if decomp("Models.CardPools", "FoolCardPool.cs").is_none() {
        return;
    }
    let mut want: Vec<String> = names_in(
        &decomp("Models.CardPools", "FoolCardPool.cs").unwrap(),
        "GenerateAllCards",
    );
    let tokens = names_in(
        &decomp("Models.CardPools", "TokenCardPool.cs").unwrap(),
        "GenerateAllCards",
    );
    // The Fool tokens are the tokens v0.113.0 adds, minus Dread's Amnesia,
    // which `TokenCardPool.cs` must still list and the engine must know.
    assert!(tokens.iter().any(|t| t == "Amnesia"));
    assert_eq!(class_name(CardId::Amnesia), "Amnesia");
    assert!(!is_fool_card(CardId::Amnesia));
    assert_eq!(
        spec_for_card_id(CardId::Amnesia).map(|s| s.id),
        Some(CardId::Amnesia)
    );
    let old_tokens = [
        "Disintegration",
        "Fuel",
        "GiantRock",
        "Luminesce",
        "MindRot",
        "MinionDiveBomb",
        "MinionSacrifice",
        "MinionStrike",
        "Shiv",
        "Sloth",
        "Soul",
        "SovereignBlade",
        "SweepingGaze",
        "WasteAway",
    ];
    let not_fool = ["Amnesia"];
    want.extend(
        tokens
            .into_iter()
            .filter(|t| !old_tokens.contains(&t.as_str()) && !not_fool.contains(&t.as_str())),
    );
    want.push("Sadness".to_string());
    let ours: Vec<String> = FOOL_CARDS
        .iter()
        .filter(|id| !CardId::V0114_ONLY.contains(id))
        .map(|&id| class_name(id))
        .collect();
    assert_eq!(ours, want);
}

#[test]
fn fool_specs_spot_checks() {
    let s = specs::strike_fool();
    assert_eq!((s.cost, s.base_damage, s.rarity), (1, 6, CardRarity::Basic));
    let d = specs::defend_fool();
    assert_eq!((d.cost, d.base_block), (1, 5));
    let f = specs::foolish_shot();
    assert_eq!(
        (f.cost, f.base_damage, f.rarity),
        (1, 15, CardRarity::Basic)
    );
    // `Brace.cs` / `RepeatShot.cs`: `HasEnergyCostX`.
    assert_eq!(specs::brace().cost, -1);
    assert_eq!(specs::repeat_shot().cost, -1);
    let sad = specs::sadness();
    assert!(sad.is_unplayable() && sad.keywords.contains(crate::cards::CardKeywords::RETAIN));
    assert_eq!(sad.rarity, CardRarity::Status);
    assert!(specs::sky_shot().exhausts());
    assert_eq!(specs::divine_shot().rarity, CardRarity::Ancient);
    // Upgrades: `StrikeFool.cs` +3 damage, `Lullaby.cs` gains Innate,
    // `BlotOutTheSky.cs` costs 1 less.
    assert_eq!(
        specs::upgrade_delta(CardId::StrikeFool)
            .unwrap()
            .base_damage,
        Some(9)
    );
    assert!(specs::upgrade_delta(CardId::LullabyCard)
        .unwrap()
        .keywords
        .is_some());
    assert_eq!(
        specs::upgrade_delta(CardId::BlotOutTheSky).unwrap().cost,
        Some(0)
    );
    for id in [
        CardId::CompostCard,
        CardId::CoveringFireCard,
        CardId::Enrapture,
        CardId::EnsembleCard,
        CardId::SelfDeprecate,
        CardId::Yoga,
    ] {
        assert!(is_multiplayer_only(id), "{id:?}");
    }
    assert!(!is_multiplayer_only(CardId::StrikeFool));
}

#[test]
fn fool_cards_rank_in_the_reshuffle_sort() {
    for id in FOOL_CARDS {
        let c = sts2_core::cards::Card {
            card_id: id,
            ..sts2_core::cards::Card::pod_defaults()
        };
        assert_eq!(
            sts2_core::cards::card_model_compare(&c, &c),
            std::cmp::Ordering::Equal
        );
    }
}

/// Every Fool card's tags match its C# `CanonicalTags` (or `override Tags`).
#[test]
fn fool_card_tags_match_the_decomp() {
    use crate::cards::status::helpers::tags_for_card_id;
    use sts2_core::cards::CardTag;
    let Some(_) = decomp("Models.CardPools", "FoolCardPool.cs") else {
        return;
    };
    // Each card against the build it exists in; a card of both builds
    // against both (the tags are build-less).
    let checks = FOOL_CARDS.iter().flat_map(|&id| {
        let mut tags = Vec::new();
        if !CardId::V0114_ONLY.contains(&id) {
            tags.push("v0.113.0");
        }
        if id.exists_in(GameVersion::V0_114_0) {
            tags.push("v0.114.0");
        }
        tags.into_iter().map(move |t| (id, t))
    });
    for (id, tag) in checks {
        let Some(src) = decomp_in(tag, "Models.Cards", &format!("{}.cs", class_name(id))) else {
            assert_eq!(tag, "v0.114.0", "{id:?}: card file");
            continue;
        };
        let decl = src
            .find("CanonicalTags =>")
            .or_else(|| src.find("IEnumerable<CardTag> Tags =>"));
        let mut want: Vec<CardTag> = Vec::new();
        if let Some(at) = decl {
            let body = &src[at..];
            let body = &body[..body.find(";\n").expect("end")];
            for (name, tag) in [
                ("Strike", CardTag::Strike),
                ("Defend", CardTag::Defend),
                ("Wah", CardTag::Wah),
                ("Cho", CardTag::Cho),
                ("Ki", CardTag::Ki),
                ("Shoot", CardTag::Shoot),
            ] {
                if body.contains(&format!("CardTag.{name}")) {
                    want.push(tag);
                }
            }
        }
        want.sort();
        let got = tags_for_card_id(id).to_vec().to_vec();
        assert_eq!(got, want, "{id:?} ({tag})");
    }
}
