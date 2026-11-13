//! The five events game build v0.113.0 adds: MuseumOfTorture (shared, act 2
//! only), ShrineOfTheWarlord and ThermaeNonupeipe (Glory),
//! WaterfallOfObviousSecrets and WuppyInTrouble (Underdocks).
//!
//! They exist only in v0.113.0's event pools (`run_room_gen::event_pools_for`),
//! so no earlier build ever reaches them. Each handler cites its event class
//! (`Models.Events/<Name>.cs`) in the v0.113.0 decomp.
//!
//! The relics they hand out are run-level only for now: their combat hooks
//! (Bath Towel's heal, Tormentous Collar's Weak, Wuppy's pet and damage
//! redirect, ...) have no `RelicKind`, so they do nothing in combat.

use sts2_core::cards::CardId;
use sts2_core::run_room_gen::AncientId;
use sts2_core::{
    EventId, EventOptionInfo, EventOutcome, EventVarSpec, PotionReward, RelicId, Reward, RunState,
};

use super::{EventOptionSpec, EventSpec};
use crate::cards::spec_index::card_pod_from_id;

/// Fallback when the grab bag runs dry — C# `RelicFactory.FallbackRelic`.
const FALLBACK_RELIC: RelicId = RelicId::Circlet;

fn not_finished(description: &str) -> EventOutcome {
    EventOutcome {
        finished: false,
        description: description.to_string(),
        effects: Vec::new(),
    }
}

fn add_card(state: &mut RunState, id: CardId) {
    if let Some(card) = card_pod_from_id(id, false) {
        state.add_card_to_deck_with_hooks(card, sts2_core::relic_effects::AddSource::EventGain);
    }
}

/// The Ancient of the act the player is in (`RunState.Act.Ancient`), when
/// room generation routed one.
fn current_act_ancient(state: &RunState) -> Option<AncientId> {
    let idx = usize::from(state.act_index.max(1) - 1);
    state.generated_rooms.get(idx)?.ancient
}

/// One option of a hand-built page.
fn info(
    option_id: &'static str,
    loc_key: &'static str,
    label: &str,
    enabled: bool,
) -> EventOptionInfo {
    EventOptionInfo {
        option_id,
        loc_key,
        vars: Vec::new(),
        title_vars: Vec::new(),
        label: label.to_string(),
        description: String::new(),
        enabled,
    }
}

const fn option(
    option_id: &'static str,
    loc_key: &'static str,
    default_label: &'static str,
    handler: fn(&mut RunState) -> EventOutcome,
) -> EventOptionSpec {
    EventOptionSpec {
        option_id,
        loc_key,
        default_label,
        default_description: "",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler,
    }
}

/// The options of `spec` whose ids are `ids`, in that order, all enabled.
fn page(spec: &EventSpec, ids: &[&str]) -> Vec<EventOptionInfo> {
    ids.iter()
        .filter_map(|id| spec.options.iter().find(|o| o.option_id == *id))
        .map(|o| info(o.option_id, o.loc_key, o.default_label, true))
        .collect()
}

// ── MuseumOfTorture ──────────────────────────────────────────────────

const MUSEUM_STREAM: &str = "MUSEUM_OF_TORTURE";
/// `MaxHpVar(10)` — Break Out's Max HP loss.
pub const MUSEUM_BREAK_OUT_MAX_HP: i32 = 10;
/// `CalculateVars`' `base.Rng.NextItem` candidates, in decomp order.
const MUSEUM_RELICS: [RelicId; 3] = [
    RelicId::ObsidianVice,
    RelicId::Thumbscrew,
    RelicId::TormentousCollar,
];

/// `IsAllowed`: `CurrentActIndex == 1`.
fn museum_allowed(state: &RunState) -> bool {
    state.act_index == 2
}

/// `CalculateVars`: the relic on display, read #1 of the event stream.
fn museum_offered_relic(state: &RunState) -> RelicId {
    let mut rng = state.rng.named(MUSEUM_STREAM);
    MUSEUM_RELICS[rng.next_int_exclusive(0, MUSEUM_RELICS.len() as i32) as usize]
}

fn museum_options(state: &RunState, spec: &EventSpec) -> Vec<EventOptionInfo> {
    match state.event_state.page {
        0 => page(spec, &["enter"]),
        1 => page(spec, &["view_display"]),
        _ => page(spec, &["take", "break_out"]),
    }
}

fn museum_enter(state: &mut RunState) -> EventOutcome {
    state.event_state.page = 1;
    not_finished("The display case is locked.")
}

fn museum_view_display(state: &mut RunState) -> EventOutcome {
    state.event_state.page = 2;
    not_finished("A relic sits behind the glass.")
}

/// `TakeRelic`: obtain the displayed relic, then one curse from
/// `CurseCardPool.Where(CanBeGeneratedByModifiers).OrderBy(Id)` drawn with
/// `PlayerRng.NonCombatGeneration.NextItem`.
fn museum_take(state: &mut RunState) -> EventOutcome {
    let relic = museum_offered_relic(state);
    state.grant_event_relic(relic);
    let pool = crate::relics::neow_relics::NEOWS_BONES_CURSE_POOL;
    let idx = state
        .rng
        .non_combat_generation
        .next_int_exclusive(0, pool.len() as i32) as usize;
    add_card(state, pool[idx]);
    EventOutcome::finished(format!("Took {} and a curse.", relic.to_python_name()))
}

fn museum_break_out(state: &mut RunState) -> EventOutcome {
    state.lose_max_hp(MUSEUM_BREAK_OUT_MAX_HP);
    EventOutcome::finished("Broke out, losing 10 Max HP.")
}

const MUSEUM_OPTIONS: &[EventOptionSpec] = &[
    option(
        "enter",
        "MUSEUM_OF_TORTURE.pages.INITIAL.options.CONTINUE",
        "Continue",
        museum_enter,
    ),
    option(
        "view_display",
        "MUSEUM_OF_TORTURE.pages.LOCKED.options.CONTINUE",
        "Continue",
        museum_view_display,
    ),
    option(
        "take",
        "MUSEUM_OF_TORTURE.pages.DISPLAY.options.TAKE",
        "Take the Relic",
        museum_take,
    ),
    option(
        "break_out",
        "MUSEUM_OF_TORTURE.pages.DISPLAY.options.BREAK_OUT",
        "Break Out",
        museum_break_out,
    ),
];

pub const MUSEUM_OF_TORTURE: EventSpec = EventSpec {
    id: EventId::MuseumOfTorture,
    python_name: "MuseumOfTorture",
    is_allowed: museum_allowed,
    generate_options: museum_options,
    options: MUSEUM_OPTIONS,
};

// ── ShrineOfTheWarlord ───────────────────────────────────────────────

/// `IsAllowed`: `!(runState.Act.Ancient is Tanx)`.
fn shrine_allowed(state: &RunState) -> bool {
    current_act_ancient(state) != Some(AncientId::Tanx)
}

fn shrine_weapon(state: &mut RunState) -> EventOutcome {
    add_card(state, CardId::MightyWeapon);
    EventOutcome::finished("Took the Mighty Weapon.")
}

fn shrine_shield(state: &mut RunState) -> EventOutcome {
    add_card(state, CardId::MightyShield);
    EventOutcome::finished("Took the Mighty Shield.")
}

/// `TakeWings`: `AddCurseToDeck<Clumsy>` first, then obtain Mighty Wings.
fn shrine_wings(state: &mut RunState) -> EventOutcome {
    add_card(state, CardId::Clumsy);
    state.grant_event_relic(RelicId::MightyWings);
    EventOutcome::finished("Took the Mighty Wings and Clumsy.")
}

const SHRINE_OPTIONS: &[EventOptionSpec] = &[
    option(
        "take_weapon",
        "SHRINE_OF_THE_WARLORD.pages.INITIAL.options.TAKE_WEAPON",
        "Take the Weapon",
        shrine_weapon,
    ),
    option(
        "take_shield",
        "SHRINE_OF_THE_WARLORD.pages.INITIAL.options.TAKE_SHIELD",
        "Take the Shield",
        shrine_shield,
    ),
    option(
        "take_wings",
        "SHRINE_OF_THE_WARLORD.pages.INITIAL.options.TAKE_WINGS",
        "Take the Wings",
        shrine_wings,
    ),
];

pub const SHRINE_OF_THE_WARLORD: EventSpec = EventSpec {
    id: EventId::ShrineOfTheWarlord,
    python_name: "ShrineOfTheWarlord",
    is_allowed: shrine_allowed,
    generate_options: super::default_option_info_from_spec,
    options: SHRINE_OPTIONS,
};

// ── ThermaeNonupeipe ─────────────────────────────────────────────────

const THERMAE_STREAM: &str = "THERMAE_NONUPEIPE";
/// `GoldVar("ThermalBathsCost", 88)`.
pub const THERMAE_BATHS_COST: i32 = 88;
/// `HpLossVar("DrinkHpLoss", 31)`.
pub const THERMAE_DRINK_HP_LOSS: i32 = 31;
/// `CardsVar("DrinkGuiltyCount", 3)`.
const THERMAE_DRINK_GUILTY: usize = 3;
/// `Nonupeipe.AllPossibleOptions` (`OptionPool` then the Beautiful Bracelet
/// option) with relic-less options and LoomingFruit removed — the
/// `CalculateVars` `NextItem` candidates, in order.
const THERMAE_DRINK_RELICS: [RelicId; 9] = [
    RelicId::BlessedAntler,
    RelicId::BrilliantScarf,
    RelicId::DelicateFrond,
    RelicId::DiamondDiadem,
    RelicId::FurCoat,
    RelicId::Glitter,
    RelicId::JewelryBox,
    RelicId::SignetRing,
    RelicId::BeautifulBracelet,
];

/// `IsAllowed`: `Gold > 175 && CurrentHp <= MaxHp * 0.75` and the act's
/// Ancient is not Nonupeipe.
fn thermae_allowed(state: &RunState) -> bool {
    // `(decimal)CurrentHp <= (decimal)MaxHp * 0.75m`, exact in integers.
    state.gold > 175
        && i64::from(state.current_hp) * 4 <= i64::from(state.max_hp) * 3
        && current_act_ancient(state) != Some(AncientId::Nonupeipe)
}

/// `CalculateVars`: the Drink the Waters relic, read #1 of the event stream.
fn thermae_drink_relic(state: &RunState) -> RelicId {
    let mut rng = state.rng.named(THERMAE_STREAM);
    THERMAE_DRINK_RELICS[rng.next_int_exclusive(0, THERMAE_DRINK_RELICS.len() as i32) as usize]
}

fn thermae_baths(state: &mut RunState) -> EventOutcome {
    state.gold = state.gold.saturating_sub(THERMAE_BATHS_COST).max(0);
    state.grant_event_relic(RelicId::BathTowel);
    EventOutcome::finished("Paid 88 gold for the thermal baths.")
}

fn thermae_oasis(state: &mut RunState) -> EventOutcome {
    state.gold = 0;
    state.grant_event_relic(RelicId::BathRobes);
    EventOutcome::finished("Paid all your gold for a private oasis.")
}

/// `DrinkWaters`: 31 unblockable damage, 3 Guilty, then the rolled relic.
fn thermae_drink(state: &mut RunState) -> EventOutcome {
    let relic = thermae_drink_relic(state);
    let _ = state.lose_hp(THERMAE_DRINK_HP_LOSS);
    for _ in 0..THERMAE_DRINK_GUILTY {
        add_card(state, CardId::Guilty);
    }
    state.grant_event_relic(relic);
    EventOutcome::finished(format!("Drank the waters: {}.", relic.to_python_name()))
}

const THERMAE_OPTIONS: &[EventOptionSpec] = &[
    EventOptionSpec {
        default_vars: &[(
            "ThermalBathsCost",
            EventVarSpec::Int(THERMAE_BATHS_COST as i64),
        )],
        ..option(
            "thermal_baths",
            "THERMAE_NONUPEIPE.pages.INITIAL.options.THERMAL_BATHS",
            "Thermal Baths",
            thermae_baths,
        )
    },
    option(
        "private_oasis",
        "THERMAE_NONUPEIPE.pages.INITIAL.options.PRIVATE_OASIS",
        "Private Oasis",
        thermae_oasis,
    ),
    EventOptionSpec {
        default_vars: &[
            (
                "DrinkHpLoss",
                EventVarSpec::Int(THERMAE_DRINK_HP_LOSS as i64),
            ),
            (
                "DrinkGuiltyCount",
                EventVarSpec::Int(THERMAE_DRINK_GUILTY as i64),
            ),
        ],
        ..option(
            "drink_waters",
            "THERMAE_NONUPEIPE.pages.INITIAL.options.DRINK_WATERS",
            "Drink the Waters",
            thermae_drink,
        )
    },
];

pub const THERMAE_NONUPEIPE: EventSpec = EventSpec {
    id: EventId::ThermaeNonupeipe,
    python_name: "ThermaeNonupeipe",
    is_allowed: thermae_allowed,
    generate_options: super::default_option_info_from_spec,
    options: THERMAE_OPTIONS,
};

// ── WaterfallOfObviousSecrets ────────────────────────────────────────

const WATERFALL_STREAM: &str = "WATERFALL_OF_OBVIOUS_SECRETS";
/// `DynamicVar("NegotiateCostKey", 70)`.
pub const WATERFALL_NEGOTIATE_COST: i32 = 70;
/// `DynamicVar("BeatUpCurseCount", 2)`.
const WATERFALL_BEAT_UP_GUILTY: usize = 2;

fn waterfall_options(state: &RunState, spec: &EventSpec) -> Vec<EventOptionInfo> {
    if state.event_state.page == 0 {
        return page(spec, &["enter"]);
    }
    let mut opts = page(spec, &["be_nice", "beat_up"]);
    // `Gold >= NegotiateCostKey` picks NEGOTIATE, else the handler-less
    // NEGOTIATE_LOCKED option.
    let can_pay = state.gold >= WATERFALL_NEGOTIATE_COST;
    opts.push(info(
        "negotiate",
        if can_pay {
            "WATERFALL_OF_OBVIOUS_SECRETS.pages.ENTER.options.NEGOTIATE"
        } else {
            "WATERFALL_OF_OBVIOUS_SECRETS.pages.ENTER.options.NEGOTIATE_LOCKED"
        },
        "Negotiate",
        can_pay,
    ));
    opts
}

fn waterfall_enter(state: &mut RunState) -> EventOutcome {
    state.event_state.page = 1;
    not_finished("A gremlin guards the waterfall.")
}

fn waterfall_be_nice(state: &mut RunState) -> EventOutcome {
    add_card(state, CardId::DingyDagger);
    EventOutcome::finished("Gained a Dingy Dagger.")
}

/// `BeatUp`: `PullNextRelicFromFront(Owner, base.Rng)` (rarity = read #1 of
/// the event stream), obtain it, then 2 Guilty.
fn waterfall_beat_up(state: &mut RunState) -> EventOutcome {
    let rarity = super::event_relic_rarity(state, WATERFALL_STREAM, 0);
    let relic = state
        .pop_relic_from_grab_bag(rarity)
        .unwrap_or(FALLBACK_RELIC);
    state.grant_event_relic(relic);
    for _ in 0..WATERFALL_BEAT_UP_GUILTY {
        add_card(state, CardId::Guilty);
    }
    EventOutcome::finished(format!("Beat up the gremlin: {}.", relic.to_python_name()))
}

/// `Negotiate`: lose 70 gold, then `CardFactory.CreateForReward(owner, 1,
/// Uniform odds, Rarity == Rare, NoUpgradeRoll)` over the character pool —
/// one `PlayerRng.Rewards` read — straight into the deck.
fn waterfall_negotiate(state: &mut RunState) -> EventOutcome {
    if state.gold < WATERFALL_NEGOTIATE_COST {
        return EventOutcome::finished("Could not afford to negotiate.");
    }
    state.gold -= WATERFALL_NEGOTIATE_COST;
    let character = state.character_id;
    let cards = crate::cards::spec_index::roll_uniform_pool_cards(
        state,
        character,
        crate::cards::CardRarity::Rare,
        1,
    );
    if let Some(card) = cards.into_iter().next() {
        state.add_card_to_deck_with_hooks(card, sts2_core::relic_effects::AddSource::EventGain);
    }
    EventOutcome::finished("Negotiated for a rare card.")
}

const WATERFALL_OPTIONS: &[EventOptionSpec] = &[
    option(
        "enter",
        "WATERFALL_OF_OBVIOUS_SECRETS.pages.INITIAL.options.ENTER",
        "Enter",
        waterfall_enter,
    ),
    option(
        "be_nice",
        "WATERFALL_OF_OBVIOUS_SECRETS.pages.ENTER.options.BE_NICE",
        "Be Nice",
        waterfall_be_nice,
    ),
    option(
        "beat_up",
        "WATERFALL_OF_OBVIOUS_SECRETS.pages.ENTER.options.BEAT_UP",
        "Beat Up",
        waterfall_beat_up,
    ),
    option(
        "negotiate",
        "WATERFALL_OF_OBVIOUS_SECRETS.pages.ENTER.options.NEGOTIATE",
        "Negotiate",
        waterfall_negotiate,
    ),
];

pub const WATERFALL_OF_OBVIOUS_SECRETS: EventSpec = EventSpec {
    id: EventId::WaterfallOfObviousSecrets,
    python_name: "WaterfallOfObviousSecrets",
    is_allowed: super::always_allowed,
    generate_options: waterfall_options,
    options: WATERFALL_OPTIONS,
};

// ── WuppyInTrouble ───────────────────────────────────────────────────

/// `DynamicVar("ImmediateWuppyHp", 60)`.
pub const WUPPY_IMMEDIATE_HP: i32 = 60;
/// `DynamicVar("CautiousWuppyHp", 30)`.
pub const WUPPY_CAUTIOUS_HP: i32 = 30;

/// `GetHpLoss`: `CurrentHp * 0.05`. `CreatureCmd.Damage` truncates the
/// decimal (`(int)damageRemaining`), so the HP actually lost is the floor.
fn wuppy_hp_loss(state: &RunState) -> i32 {
    state.current_hp.max(0) * 5 / 100
}

/// `IsAllowed`: `CurrentHp > CurrentHp * 0.05` (any living player) and no
/// event pet yet.
fn wuppy_allowed(state: &RunState) -> bool {
    state.current_hp > 0 && !state.has_event_pet()
}

/// Obtain Wuppy with `max_hp` (`wuppy.MaxHp = ...`), kept in the relic
/// counter.
fn obtain_wuppy(state: &mut RunState, max_hp: i32) {
    state.grant_event_relic(RelicId::Wuppy);
    state.relic_counters.insert(RelicId::Wuppy, max_hp);
}

/// `HelpImmediately`: Injury, then Wuppy at 60 HP.
fn wuppy_help(state: &mut RunState) -> EventOutcome {
    add_card(state, CardId::Injury);
    obtain_wuppy(state, WUPPY_IMMEDIATE_HP);
    EventOutcome::finished("Helped Wuppy at once.")
}

/// `BeCautious`: lose 5% current HP (unblockable), then Wuppy at 30 HP.
fn wuppy_cautious(state: &mut RunState) -> EventOutcome {
    let loss = wuppy_hp_loss(state);
    let _ = state.lose_hp(loss);
    obtain_wuppy(state, WUPPY_CAUTIOUS_HP);
    EventOutcome::finished("Helped Wuppy carefully.")
}

/// `Ignore`: one `PlayerRng.Rewards.NextItem` over the character ++ shared
/// potion pool, offered as a populated `PotionReward`.
fn wuppy_ignore(state: &mut RunState) -> EventOutcome {
    if let Some(kind) =
        crate::potions::all::roll_event_potion_full_pool(&mut state.rng.rewards, state.character_id)
    {
        state
            .pending_rewards
            .push(Reward::Potion(PotionReward::with_kind(kind)));
    }
    EventOutcome::finished("Left Wuppy to its fate.")
}

const WUPPY_OPTIONS: &[EventOptionSpec] = &[
    EventOptionSpec {
        default_vars: &[(
            "ImmediateWuppyHp",
            EventVarSpec::Int(WUPPY_IMMEDIATE_HP as i64),
        )],
        ..option(
            "help_immediately",
            "WUPPY_IN_TROUBLE.pages.INITIAL.options.HELP_IMMEDIATELY",
            "Help Immediately",
            wuppy_help,
        )
    },
    EventOptionSpec {
        vars_fn: Some(|s| {
            vec![
                ("CautiousWuppyHp", i64::from(WUPPY_CAUTIOUS_HP).into()),
                ("HpLoss", i64::from(wuppy_hp_loss(s)).into()),
            ]
        }),
        ..option(
            "be_cautious",
            "WUPPY_IN_TROUBLE.pages.INITIAL.options.BE_CAUTIOUS",
            "Be Cautious",
            wuppy_cautious,
        )
    },
    option(
        "ignore",
        "WUPPY_IN_TROUBLE.pages.INITIAL.options.IGNORE",
        "Ignore",
        wuppy_ignore,
    ),
];

pub const WUPPY_IN_TROUBLE: EventSpec = EventSpec {
    id: EventId::WuppyInTrouble,
    python_name: "WuppyInTrouble",
    is_allowed: wuppy_allowed,
    generate_options: super::default_option_info_from_spec,
    options: WUPPY_OPTIONS,
};

#[cfg(test)]
mod tests {
    use super::*;
    use sts2_core::GameVersion;

    fn state_v0113() -> RunState {
        let mut rs = RunState::new();
        rs.game_version = GameVersion::V0_113_0;
        rs.rng.game_version = GameVersion::V0_113_0;
        rs
    }

    #[test]
    fn museum_is_act_two_only() {
        let mut rs = state_v0113();
        for (act, allowed) in [(1, false), (2, true), (3, false)] {
            rs.act_index = act;
            assert_eq!(museum_allowed(&rs), allowed, "act {act}");
        }
    }

    #[test]
    fn museum_take_draws_one_non_combat_generation_curse() {
        let mut rs = state_v0113();
        rs.direct_event_relic_grant = true;
        let before = rs.rng.non_combat_generation.counter();
        let deck = rs.master_deck.len();
        let relic = museum_offered_relic(&rs);
        let _ = museum_take(&mut rs);
        assert_eq!(rs.rng.non_combat_generation.counter(), before + 1);
        assert_eq!(rs.master_deck.len(), deck + 1);
        assert!(rs.relics.contains(&relic));
    }

    #[test]
    fn thermae_needs_gold_and_missing_hp() {
        let mut rs = state_v0113();
        rs.max_hp = 80;
        rs.current_hp = 60; // exactly 75%
        rs.gold = 176;
        assert!(thermae_allowed(&rs));
        rs.current_hp = 61;
        assert!(!thermae_allowed(&rs));
        rs.current_hp = 60;
        rs.gold = 175;
        assert!(!thermae_allowed(&rs));
    }

    #[test]
    fn waterfall_negotiate_locks_below_cost() {
        let mut rs = state_v0113();
        rs.event_state.page = 1;
        rs.gold = WATERFALL_NEGOTIATE_COST - 1;
        let opts = waterfall_options(&rs, &WATERFALL_OF_OBVIOUS_SECRETS);
        assert_eq!(opts.len(), 3);
        assert!(!opts[2].enabled);
        rs.gold = WATERFALL_NEGOTIATE_COST;
        assert!(waterfall_options(&rs, &WATERFALL_OF_OBVIOUS_SECRETS)[2].enabled);
    }

    #[test]
    fn wuppy_cautious_loses_floor_five_percent() {
        let mut rs = state_v0113();
        rs.direct_event_relic_grant = true;
        rs.max_hp = 80;
        rs.current_hp = 79; // 3.95 -> 3
        let _ = wuppy_cautious(&mut rs);
        assert_eq!(rs.current_hp, 76);
        assert_eq!(
            rs.relic_counters.get(&RelicId::Wuppy),
            Some(&WUPPY_CAUTIOUS_HP)
        );
        assert!(!wuppy_allowed(&rs), "Wuppy is an event pet");
    }
}
