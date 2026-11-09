//! String tables shared by the emitters and their hosts.
//!
//! These are the vocabularies the Python surface has always spoken —
//! `phase()` → `"map_choice"`, `take_action()` → `"advanced"`, the
//! `pending_choice()["kind"]` discriminant, the `full_map()` node types — and
//! the `PyO3` and workbench hosts read them from here rather than each
//! carrying its own copy.

use sts2_core::cards::CardType;
use sts2_core::events::EventEncounter;
use sts2_core::map::MapPointType;
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{ActionResult, Phase};
use sts2_core::runstate::RoomType;
use sts2_core::ChoiceKind;

/// `RunManager.phase()` — lowercase `snake_case`.
#[must_use]
pub const fn phase_str(p: Phase) -> &'static str {
    match p {
        Phase::MapChoice => "map_choice",
        Phase::Combat => "combat",
        Phase::CardReward => "card_reward",
        Phase::BossRelic => "boss_relic",
        Phase::Shop => "shop",
        Phase::RestSite => "rest_site",
        Phase::Event => "event",
        Phase::Treasure => "treasure",
        Phase::RunOver => "run_over",
    }
}

/// `RunManager.current_room()["room_type"]` — lowercase `snake_case`.
#[must_use]
pub const fn room_type_str(rt: RoomType) -> &'static str {
    match rt {
        RoomType::Unknown => "unknown",
        RoomType::Monster => "monster",
        RoomType::Elite => "elite",
        RoomType::Boss => "boss",
        RoomType::Shop => "shop",
        RoomType::RestSite => "rest_site",
        RoomType::Treasure => "treasure",
        RoomType::Event => "event",
    }
}

/// `RunManager.take_action()`'s return string.
#[must_use]
pub const fn result_str(r: &ActionResult) -> &'static str {
    match r {
        ActionResult::Advanced => "advanced",
        ActionResult::InvalidForPhase => "invalid_for_phase",
        ActionResult::NotImplemented => "not_implemented",
        ActionResult::RunOver => "run_over",
        // Bead `sts2-rs-8he0t`: typed shop-failure variants surface
        // through a single string for backward compatibility; harness
        // code that needs the inner reason matches on the
        // `ActionResult` directly (Rust callers) or inspects the
        // returned action dict on the Python side.
        ActionResult::ShopPurchaseFailed(_) => "shop_purchase_failed",
        ActionResult::ShopRemoveCardFailed(_) => "shop_remove_card_failed",
        ActionResult::InvalidTarget => "invalid_target",
        ActionResult::IllegalPlay => "illegal_play",
        _ => "unknown",
    }
}

/// Parse the `character=` constructor argument. Accepts the `PascalCase`
/// name or its lowercase form.
///
/// # Errors
///
/// The message the Python constructor raises as `ValueError` for an unknown
/// name.
pub fn character_from_str(s: &str) -> Result<CharacterId, String> {
    match s {
        "Ironclad" | "ironclad" => Ok(CharacterId::Ironclad),
        "Silent" | "silent" => Ok(CharacterId::Silent),
        "Defect" | "defect" => Ok(CharacterId::Defect),
        "Regent" | "regent" => Ok(CharacterId::Regent),
        "Necrobinder" | "necrobinder" => Ok(CharacterId::Necrobinder),
        other => Err(format!(
            "unknown character {other:?} — expected one of \
             Ironclad/Silent/Defect/Regent/Necrobinder"
        )),
    }
}

/// `RunManager.character` — `PascalCase`.
#[must_use]
pub const fn character_str(c: CharacterId) -> &'static str {
    match c {
        CharacterId::Ironclad => "Ironclad",
        CharacterId::Silent => "Silent",
        CharacterId::Defect => "Defect",
        CharacterId::Regent => "Regent",
        CharacterId::Necrobinder => "Necrobinder",
    }
}

/// Map a [`ChoiceKind`] to a stable kebab/snake-case string for the
/// Rust-side `kind` field. Python's `PendingCardChoice` has no
/// discriminant (the flavor is encoded in the `prompt` string), so
/// these names are Rust-port additions for callers that want to branch
/// without parsing prompts.
#[must_use]
pub const fn choice_kind_python_name(k: ChoiceKind) -> &'static str {
    match k {
        ChoiceKind::CardSelect => "card_select",
        ChoiceKind::PutBack => "put_back",
        ChoiceKind::ExhaustPile => "exhaust_pile",
        ChoiceKind::OrbSlot => "orb_slot",
        ChoiceKind::EnemyTarget => "enemy_target",
        ChoiceKind::AddGeneratedToHand => "add_generated_to_hand",
        ChoiceKind::AddGeneratedRetainedToHand => "add_generated_retained_to_hand",
        ChoiceKind::AddGeneratedFreeToHand => "add_generated_free_to_hand",
        ChoiceKind::PaelsToothStash => "paels_tooth_stash",
        ChoiceKind::UpgradeInHand => "upgrade_in_hand",
        ChoiceKind::AddFromDrawByType => "add_from_draw_by_type",
        ChoiceKind::AddFromDiscardFree => "add_from_discard_free",
        ChoiceKind::DiscardFromHand => "discard_from_hand",
        ChoiceKind::ExhaustFromHand => "exhaust_from_hand",
        ChoiceKind::DiscardFromHandAndDraw => "discard_from_hand_and_draw",
        ChoiceKind::MakeFreeInHand => "make_free_in_hand",
        ChoiceKind::MonsterCurseSelect => "monster_curse_select",
        ChoiceKind::AddFromDrawToHand => "add_from_draw_to_hand",
        ChoiceKind::ForegoneConclusionFromDraw => "foregone_conclusion_from_draw",
        ChoiceKind::AddFromDiscardToHand => "add_from_discard_to_hand",
        ChoiceKind::ApplyRetainInHand => "apply_retain_in_hand",
        ChoiceKind::ApplySlyInHand => "apply_sly_in_hand",
        ChoiceKind::ApplyEtherealInHand => "apply_ethereal_in_hand",
        ChoiceKind::TransformInHandToMinionStrike => "transform_in_hand_to_minion_strike",
        ChoiceKind::TransformInHandToMinionSacrifice => "transform_in_hand_to_minion_sacrifice",
        ChoiceKind::TransfigureCardInHand => "transfigure_card_in_hand",
        ChoiceKind::TransformDrawCardToSoul => "transform_draw_card_to_soul",
        ChoiceKind::TransformDrawCardsToMinionDiveBomb => {
            "transform_draw_cards_to_minion_dive_bomb"
        }
        ChoiceKind::CloneChosenCardNextTurn => "clone_chosen_card_next_turn",
        ChoiceKind::RetainAtTurnEnd => "retain_at_turn_end",
        ChoiceKind::AutoPlaySkillFromHand => "auto_play_skill_from_hand",
    }
}

/// Map a [`CardType`] to the title-cased string emitted by Python's
/// `_CARD_TYPE_NAMES` table (`bridge_replay.py:43-50`). Duplicate of
/// the private helper in `sts2_core::parity` — kept here so the wire
/// avoids a `pub(crate)` carve-out in core just for this.
#[must_use]
pub const fn card_type_python_name(t: CardType) -> &'static str {
    match t {
        CardType::Attack => "Attack",
        CardType::Skill => "Skill",
        CardType::Power => "Power",
        CardType::Status => "Status",
        CardType::Curse => "Curse",
        CardType::Quest => "Quest",
    }
}

/// Lowercase `snake_case` map point types — the `act_map()` corpus-tooling
/// schema.
#[must_use]
pub const fn map_point_type_str(t: MapPointType) -> &'static str {
    match t {
        MapPointType::Unassigned => "unassigned",
        MapPointType::Unknown => "unknown",
        MapPointType::Monster => "monster",
        MapPointType::Elite => "elite",
        MapPointType::Boss => "boss",
        MapPointType::Shop => "shop",
        MapPointType::RestSite => "rest_site",
        MapPointType::Treasure => "treasure",
        MapPointType::Ancient => "ancient",
    }
}

/// `PascalCase` room-type strings keyed by the web UI's `NODE` glyph table
/// (`sts2-pycli/web/src/components/MapView.tsx`). Distinct from
/// [`map_point_type_str`] (lowercase snake, used by `act_map()` corpus
/// tooling): the `full_map()` schema must match the C# `PointType.ToString()`
/// casing the UI looks up.
#[must_use]
pub const fn map_point_type_ui_str(t: MapPointType) -> &'static str {
    match t {
        MapPointType::Unassigned => "Unassigned",
        MapPointType::Unknown => "Unknown",
        MapPointType::Monster => "Monster",
        MapPointType::Elite => "Elite",
        MapPointType::Boss => "Boss",
        MapPointType::Shop => "Shop",
        MapPointType::RestSite => "RestSite",
        MapPointType::Treasure => "Treasure",
        MapPointType::Ancient => "Ancient",
    }
}

/// The C# id of an event's fight (`ModelId.Entry`, the slugged class name of
/// the `EncounterModel` the event passes to `EnterCombatWithoutExitingEvent`),
/// as `CombatState.encounter_id` reports it during that fight (GH #1097), in
/// game build v0.107.1, where the three `BattlewornDummy` settings share one
/// encounter class. [`event_encounter_id_for`] takes the build.
#[must_use]
pub const fn event_encounter_id(e: EventEncounter) -> &'static str {
    match e {
        EventEncounter::BattlewornDummy(_) => "BATTLEWORN_DUMMY_EVENT_ENCOUNTER",
        EventEncounter::MysteriousKnight => "MYSTERIOUS_KNIGHT_EVENT_ENCOUNTER",
        EventEncounter::FakeMerchant => "FAKE_MERCHANT_EVENT_ENCOUNTER",
        EventEncounter::DenseVegetation => "DENSE_VEGETATION_EVENT_ENCOUNTER",
    }
}

/// [`event_encounter_id`] in game build `version` (#1181). v0.111.0 gives
/// each `BattlewornDummy` setting its own encounter class
/// (`BattlewornDummyEventV1Encounter` … `V3`, `BattlewornDummy.cs:54-66`),
/// slugged like `BattleFriendV1` → `BATTLE_FRIEND_V1`. The other event
/// fights keep one id in both builds.
#[must_use]
pub fn event_encounter_id_for(e: EventEncounter, version: sts2_core::GameVersion) -> &'static str {
    use sts2_core::events::BattlewornDummySetting as S;
    match e {
        EventEncounter::BattlewornDummy(setting) if version >= sts2_core::GameVersion::V0_111_0 => {
            match setting {
                S::Setting1 => "BATTLEWORN_DUMMY_EVENT_V1_ENCOUNTER",
                S::Setting2 => "BATTLEWORN_DUMMY_EVENT_V2_ENCOUNTER",
                S::Setting3 => "BATTLEWORN_DUMMY_EVENT_V3_ENCOUNTER",
            }
        }
        other => event_encounter_id(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn battleworn_dummy_has_one_encounter_per_setting_from_v0111() {
        use sts2_core::events::BattlewornDummySetting as S;
        use sts2_core::GameVersion;
        for (setting, id) in [
            (S::Setting1, "BATTLEWORN_DUMMY_EVENT_V1_ENCOUNTER"),
            (S::Setting2, "BATTLEWORN_DUMMY_EVENT_V2_ENCOUNTER"),
            (S::Setting3, "BATTLEWORN_DUMMY_EVENT_V3_ENCOUNTER"),
        ] {
            let e = EventEncounter::BattlewornDummy(setting);
            assert_eq!(event_encounter_id_for(e, GameVersion::V0_111_0), id);
            assert_eq!(
                event_encounter_id_for(e, GameVersion::V0_107_1),
                "BATTLEWORN_DUMMY_EVENT_ENCOUNTER"
            );
        }
        assert_eq!(
            event_encounter_id_for(EventEncounter::FakeMerchant, GameVersion::V0_111_0),
            "FAKE_MERCHANT_EVENT_ENCOUNTER"
        );
    }

    #[test]
    fn ui_type_strings_are_pascalcase_for_ui_node_table() {
        assert_eq!(map_point_type_ui_str(MapPointType::Monster), "Monster");
        assert_eq!(map_point_type_ui_str(MapPointType::Elite), "Elite");
        assert_eq!(map_point_type_ui_str(MapPointType::Boss), "Boss");
        assert_eq!(map_point_type_ui_str(MapPointType::Shop), "Shop");
        // RestSite, not rest_site — UI NODE table key (MapView.tsx).
        assert_eq!(map_point_type_ui_str(MapPointType::RestSite), "RestSite");
        assert_eq!(map_point_type_ui_str(MapPointType::Treasure), "Treasure");
        assert_eq!(map_point_type_ui_str(MapPointType::Ancient), "Ancient");
        assert_eq!(map_point_type_ui_str(MapPointType::Unknown), "Unknown");
        assert_eq!(
            map_point_type_ui_str(MapPointType::Unassigned),
            "Unassigned"
        );
    }

    #[test]
    fn character_names_round_trip() {
        for c in [
            CharacterId::Ironclad,
            CharacterId::Silent,
            CharacterId::Defect,
            CharacterId::Regent,
            CharacterId::Necrobinder,
        ] {
            assert_eq!(character_from_str(character_str(c)), Ok(c));
            assert_eq!(character_from_str(&character_str(c).to_lowercase()), Ok(c));
        }
        assert!(character_from_str("Watcher").is_err());
    }

    /// The ids are the C# encounter class names (`BattlewornDummyEventEncounter`
    /// and its siblings), slugged the way `ModelId.Entry` slugs every encounter
    /// (`SlimesWeak` → `SLIMES_WEAK`).
    #[test]
    fn event_encounter_ids_are_the_slugged_class_names() {
        use sts2_core::events::BattlewornDummySetting;
        assert_eq!(
            event_encounter_id(EventEncounter::BattlewornDummy(
                BattlewornDummySetting::Setting2
            )),
            "BATTLEWORN_DUMMY_EVENT_ENCOUNTER"
        );
        assert_eq!(
            event_encounter_id(EventEncounter::MysteriousKnight),
            "MYSTERIOUS_KNIGHT_EVENT_ENCOUNTER"
        );
        assert_eq!(
            event_encounter_id(EventEncounter::FakeMerchant),
            "FAKE_MERCHANT_EVENT_ENCOUNTER"
        );
        assert_eq!(
            event_encounter_id(EventEncounter::DenseVegetation),
            "DENSE_VEGETATION_EVENT_ENCOUNTER"
        );
    }
}
