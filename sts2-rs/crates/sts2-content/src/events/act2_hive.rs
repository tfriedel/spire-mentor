//! Act2-pool events wave A (alphabetical first 10 of 19). Bead
//! `sts2-rs-6ynt.4.6`.
//!
//! Port pin:
//! `sts2-rl-agent@01c094d:sts2_env/events/act2.py`. Wave B
//! (RelicTrader..WhisperingHollow, 9 events) lands via `sts2-rs-6ynt.4.7`.
//!
//! ## Deferral conventions
//!
//! Some handler legs short-circuit to a `finished: true` outcome whose
//! description names the deferred surface, matching the wave-A shared-pool
//! precedent in [`super::shared`]. The remaining deferrals are now tracked
//! under umbrella `sts2-rs-m3n0g` (the old `6ynt.4.*` tokens below were
//! never filed as beads).
//!
//! **Status corrected 2026-06-26** (verified against current bodies):
//! - `6ynt.4.act-gate` — **DONE.** Every `is_allowed` predicate is wired
//!   (`crystal_sphere_allowed`, `allowed_act_2_only`, `fake_merchant_allowed`,
//!   `allowed_act_2_plus`, `ranwid_the_elder_allowed`, …). The per-handler
//!   "deferred to 6ynt.4.act-gate" notes below are stale.
//! - `6ynt.4.deck-add` — the deck-mutation SURFACE exists
//!   (`pending_deck_{removals,transforms,upgrades,enchants}`) and several
//!   handlers use it; the leftover stubs just aren't wired (`sts2-rs-m3n0g.2`).
//! - `6ynt.4.multi-page` — genuinely missing surface (`sts2-rs-m3n0g.3`):
//!   DollRoom 2-doll pick, EndlessConveyor belt loop, FakeMerchant inventory.
//! - `6ynt.4.rng` — event-side relic-pool roll still missing
//!   (`sts2-rs-m3n0g.1`): random doll/relic, uncommon potion, relic-trade pool.
//! - `6ynt.4.potion-add` — `Reward::Potion` LANDED. PotionCourier grab/ransack
//!   are now wired (`sts2-rs-m3n0g.4`); FakeMerchant.throw_foul still routes
//!   through the event→combat bridge (`sts2-rs-m3n0g.5`).

use sts2_core::events::{EventCombatRequest, EventEncounter};
use sts2_core::{
    EventId, EventOptionInfo, EventOutcome, EventVarSpec, RelicId, RelicReward, Reward, RunState,
};

use super::{
    allowed_act_2_only, allowed_act_2_plus, always_allowed, any_enchantable_deck_card,
    crystal_sphere_allowed, endless_conveyor_allowed, fake_merchant_allowed,
    luminous_choir_allowed, morphic_grove_allowed, ranwid_the_elder_allowed, EventOptionSpec,
    EventSpec,
};

// ── CrystalSphere ────────────────────────────────────────────────────

/// `CrystalSphere`'s `Id.Entry` — names its event Rng.
const CRYSTAL_SPHERE_STREAM: &str = "CRYSTAL_SPHERE";

/// `_uncoverFutureCost` (`CrystalSphere.cs:23`): the cost before the roll.
pub const CRYSTAL_SPHERE_BASE_COST: i32 = 50;

/// `_uncoverFutureProphesizeCount` (`CrystalSphere.cs:29`): divinations
/// bought by Uncover Future.
const CRYSTAL_SPHERE_UNCOVER_DIVINATIONS: u32 = 3;

/// `_paymentPlanCount` (`CrystalSphere.cs:31`): divinations bought by the
/// Payment Plan.
const CRYSTAL_SPHERE_PAYMENT_PLAN_DIVINATIONS: u32 = 6;

/// The event Rng after `CalculateVars`, plus the Uncover Future cost it
/// rolled: `UncoverFutureCost.BaseValue += Rng.NextInt(1, 50)`
/// (`CrystalSphere.cs:44-47`), so 51..=99. `BeginEvent` builds the Rng and
/// runs `CalculateVars` before any option is shown, and nothing else draws
/// from it until an option is taken, so re-deriving it is exact.
fn crystal_sphere_rng_and_cost(state: &RunState) -> (sts2_core::dotnet_rng::Rng, i32) {
    let mut rng = state.rng.named(CRYSTAL_SPHERE_STREAM);
    let cost = CRYSTAL_SPHERE_BASE_COST + rng.next_int_exclusive(1, 50);
    (rng, cost)
}

/// The rolled Uncover Future cost (`UncoverFutureCost`).
#[must_use]
pub fn crystal_sphere_pay_cost(state: &RunState) -> i32 {
    crystal_sphere_rng_and_cost(state).1
}

fn crystal_sphere_generate_options(state: &RunState, spec: &EventSpec) -> Vec<EventOptionInfo> {
    let cost = crystal_sphere_pay_cost(state);
    // C# offers both options unconditionally. `IsAllowed` needs 100 gold and
    // the cost is at most 99, so Uncover Future is always affordable.
    let mut options = super::default_option_info_from_spec(state, spec);
    if let Some(pay) = options.first_mut() {
        pay.label = format!("Uncover Future ({cost}g)");
        pay.description = format!("Spend {cost} gold for 3 divinations.");
    }
    options
}

/// `UncoverFuture` (`CrystalSphere.cs:66-72`): `LoseGold(cost, Spent)`, then
/// a 3-divination minigame on the event Rng.
fn crystal_sphere_pay(state: &mut RunState) -> EventOutcome {
    let (mut rng, cost) = crystal_sphere_rng_and_cost(state);
    state.gold = state.gold.saturating_sub(cost).max(0);
    super::crystal_sphere::play_headless_and_offer(
        state,
        &mut rng,
        CRYSTAL_SPHERE_UNCOVER_DIVINATIONS,
    );
    EventOutcome::finished(format!("Paid {cost} gold and gazed into the sphere."))
}

/// `PaymentPlan` (`CrystalSphere.cs:74-80`): `AddCurseToDeck<Debt>`, then a
/// 6-divination minigame on the event Rng.
fn crystal_sphere_debt(state: &mut RunState) -> EventOutcome {
    use sts2_core::cards::CardId;
    let (mut rng, _cost) = crystal_sphere_rng_and_cost(state);
    super::crystal_sphere::add_curse_to_deck(state, CardId::Debt);
    super::crystal_sphere::play_headless_and_offer(
        state,
        &mut rng,
        CRYSTAL_SPHERE_PAYMENT_PLAN_DIVINATIONS,
    );
    EventOutcome::finished("Took the Payment Plan and gazed into the sphere.")
}

/// The part of a CrystalSphere option the player can see before taking it:
/// the gold paid, or the Debt curse. The divination haul is rolled on the
/// event Rng only once an option is taken, so the preview must not play the
/// minigame — doing so handed the policy the exact gold / potions / cards it
/// was about to win. Returns `false` for an unknown `option_id`.
pub(super) fn crystal_sphere_apply_visible(state: &mut RunState, option_id: &str) -> bool {
    match option_id {
        "pay" => {
            let cost = crystal_sphere_pay_cost(state);
            state.gold = state.gold.saturating_sub(cost).max(0);
            true
        }
        "debt" => {
            super::crystal_sphere::add_curse_to_deck(state, sts2_core::cards::CardId::Debt);
            true
        }
        _ => false,
    }
}

const CRYSTAL_SPHERE_OPTIONS: &[EventOptionSpec] = &[
    EventOptionSpec {
        option_id: "pay",
        loc_key: "CRYSTAL_SPHERE.pages.INITIAL.options.UNCOVER_FUTURE",
        default_label: "Uncover Future",
        default_description: "Pay 51-99 gold for 3 divinations.",
        default_vars: &[("UncoverFutureProphesizeCount", EventVarSpec::Int(3))],
        vars_fn: Some(|s| {
            vec![(
                "UncoverFutureCost",
                sts2_core::EventVar::Int(i64::from(crystal_sphere_pay_cost(s))),
            )]
        }),
        title_vars_fn: None,
        handler: crystal_sphere_pay,
    },
    EventOptionSpec {
        option_id: "debt",
        loc_key: "CRYSTAL_SPHERE.pages.INITIAL.options.PAYMENT_PLAN",
        default_label: "Payment Plan",
        default_description: "6 divinations, gain Debt curse.",
        default_vars: &[
            ("CurseTitle", EventVarSpec::Text("Debt")),
            ("PaymentPlanCount", EventVarSpec::Int(6)),
        ],
        vars_fn: None,
        title_vars_fn: None,
        handler: crystal_sphere_debt,
    },
];

/// `CrystalSphere` (`Models.Events/CrystalSphere.cs`; identical in v0.107.1
/// and v0.111.0). The minigame lives in [`super::crystal_sphere`].
pub const CRYSTAL_SPHERE: EventSpec = EventSpec {
    id: EventId::CrystalSphere,
    python_name: "CrystalSphere",
    is_allowed: crystal_sphere_allowed,
    generate_options: crystal_sphere_generate_options,
    options: CRYSTAL_SPHERE_OPTIONS,
};

// ── DollRoom ─────────────────────────────────────────────────────────

/// `DollRoom::take_time` HP cost. C# `DamageVar("TakeTimeHpLoss", 5)`.
pub const DOLL_ROOM_TAKE_TIME_DAMAGE: i32 = 5;
/// `DollRoom::examine` HP cost. C# `DamageVar("ExamineHpLoss", 15)`.
pub const DOLL_ROOM_EXAMINE_DAMAGE: i32 = 15;

/// `DollRoom`'s `Id.Entry` — names its own (non-shared) event RNG stream.
const DOLL_ROOM_STREAM: &str = "DOLL_ROOM";

/// The three dolls in C# `_dolls` declaration order (`DollRoom.cs:41`).
/// `ChooseRandom` draws `Rng.NextItem(_dolls)` over this order; the
/// multi-page legs `StableShuffle` it. Each entry is `(relic, option_id)`.
const DOLL_ROOM_DOLLS: [(RelicId, &str); 3] = [
    (RelicId::DaughterOfTheWind, "doll_daughter"),
    (RelicId::MrStruggles, "doll_mr_struggles"),
    (RelicId::BingBong, "doll_bingbong"),
];

/// Build `DollRoom`'s own event RNG stream. C# `IsShared == false` (default),
/// so the seed folds in the single-player player term and `hash("DOLL_ROOM")`
/// — same construction as [`super::LUMINOUS_CHOIR`]. v0.107.1 additive player
/// term is 0 (`EventModel.cs:238` GetPlayerSlotIndex == 0; v0.103.2 used
/// `Owner.NetId == 1`, `EventModel.cs:193`).
fn doll_room_rng(state: &RunState) -> sts2_core::dotnet_rng::Rng {
    state.rng.named(DOLL_ROOM_STREAM)
}

/// Grant the doll relic directly (C# `ChooseDollAndShowDescription` →
/// `RelicCmd.Obtain`) and finish the event.
fn doll_room_grant(state: &mut RunState, relic: RelicId) -> EventOutcome {
    state.grant_event_relic(relic);
    EventOutcome::finished("Took a doll home.")
}

/// `ChooseRandom`: `Rng.NextItem(_dolls)` → obtain that doll, finish.
fn doll_room_random(state: &mut RunState) -> EventOutcome {
    let mut rng = doll_room_rng(state);
    let idx = rng.next_int_exclusive(0, DOLL_ROOM_DOLLS.len() as i32) as usize;
    doll_room_grant(state, DOLL_ROOM_DOLLS[idx].0)
}

/// Lose `damage` HP, then `_dolls.StableShuffle(Rng).Take(take)` → offer those
/// dolls on page 1 (NOT finished, so the run-loop stays in `Phase::Event`).
///
/// C# `StableShuffle` sorts by `relic.CompareTo` (the id entry) before the
/// Fisher-Yates pass, and so does this, in both builds (GH #806; the old
/// [`DOLL_ROOM_DOLLS`] declaration-order approximation is gone).
fn doll_room_open_pick(state: &mut RunState, damage: i32, take: usize) -> EventOutcome {
    state.lose_hp(damage);
    let mut codes: Vec<i16> = (0..DOLL_ROOM_DOLLS.len() as i16).collect();
    let mut rng = doll_room_rng(state);
    // `StableShuffle` sorts the dolls by `relic.CompareTo`, i.e.
    // `ModelId.CompareTo` = ordinal `Entry` order (decomp
    // `Models.Events/DollRoom.cs:28`, `Models/ModelId.cs:43`, identical in
    // v0.107.1 and v0.111.0): BING_BONG < DAUGHTER_OF_THE_WIND < MR_STRUGGLES.
    let entry = |c: i16| DOLL_ROOM_DOLLS[c as usize].0.to_python_name();
    rng.stable_shuffle_by(&mut codes, &|a: i16, b: i16| entry(a).cmp(entry(b)));
    codes.truncate(take);
    state.event_state.scratch = codes;
    state.event_state.page = 1;
    EventOutcome {
        finished: false,
        description: "Choose a doll.".to_string(),
        effects: Vec::new(),
    }
}

fn doll_room_take_time(state: &mut RunState) -> EventOutcome {
    doll_room_open_pick(state, DOLL_ROOM_TAKE_TIME_DAMAGE, 2)
}

fn doll_room_examine(state: &mut RunState) -> EventOutcome {
    doll_room_open_pick(state, DOLL_ROOM_EXAMINE_DAMAGE, 3)
}

fn doll_room_pick_daughter(state: &mut RunState) -> EventOutcome {
    doll_room_grant(state, RelicId::DaughterOfTheWind)
}
fn doll_room_pick_mr_struggles(state: &mut RunState) -> EventOutcome {
    doll_room_grant(state, RelicId::MrStruggles)
}
fn doll_room_pick_bingbong(state: &mut RunState) -> EventOutcome {
    doll_room_grant(state, RelicId::BingBong)
}

/// Multi-page presenter: page 0 = the three approach options; page 1 = the
/// offered dolls (the `StableShuffle`d subset stored in `event_state.scratch`).
fn doll_room_generate_options(state: &RunState, spec: &EventSpec) -> Vec<EventOptionInfo> {
    if state.event_state.page == 1 {
        return state
            .event_state
            .scratch
            .iter()
            .map(|&code| {
                let (relic, option_id) = DOLL_ROOM_DOLLS[code as usize];
                EventOptionInfo {
                    option_id,
                    // The game names this row after the doll that landed in
                    // the slot (`DOLL_ROOM.pages.TAKE.options.TAKE` with a
                    // `{RelicName}` var), so no static key fits. The relic's
                    // own title carries the text a player needs.
                    loc_key: "",
                    vars: Vec::new(),
                    title_vars: Vec::new(),
                    label: format!("{relic:?}"),
                    description: "Take this doll.".to_string(),
                    enabled: true,
                }
            })
            .collect();
    }
    // Page 0 is only the three approaches (C# `GenerateInitialOptions`,
    // `DollRoom.cs:84`, same in both builds). The static list also holds
    // the three page-1 doll picks (so `resolve_by_index` can key their
    // handlers); presenting the whole list offered 6 options instead of 3.
    let mut opts = super::default_option_info_from_spec(state, spec);
    opts.truncate(DOLL_ROOM_INITIAL_OPTION_COUNT);
    opts
}

/// Page-0 option count: Random / Take Some Time / Examine.
const DOLL_ROOM_INITIAL_OPTION_COUNT: usize = 3;

const DOLL_ROOM_OPTIONS: &[EventOptionSpec] = &[
    // ── page 0: the three approaches ──
    EventOptionSpec {
        option_id: "random",
        loc_key: "DOLL_ROOM.pages.INITIAL.options.RANDOM",
        default_label: "Random",
        default_description: "Get a random doll relic.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: doll_room_random,
    },
    EventOptionSpec {
        option_id: "take_time",
        loc_key: "DOLL_ROOM.pages.INITIAL.options.TAKE_SOME_TIME",
        default_label: "Take Some Time",
        default_description: "Take 5 damage, choose from 2 dolls.",
        default_vars: &[(
            "TakeTimeHpLoss",
            EventVarSpec::Int(DOLL_ROOM_TAKE_TIME_DAMAGE as i64),
        )],
        vars_fn: None,
        title_vars_fn: None,
        handler: doll_room_take_time,
    },
    EventOptionSpec {
        option_id: "examine",
        loc_key: "DOLL_ROOM.pages.INITIAL.options.EXAMINE",
        default_label: "Examine",
        default_description: "Take 15 damage, choose from 3 dolls.",
        default_vars: &[(
            "ExamineHpLoss",
            EventVarSpec::Int(DOLL_ROOM_EXAMINE_DAMAGE as i64),
        )],
        vars_fn: None,
        title_vars_fn: None,
        handler: doll_room_examine,
    },
    // ── page 1: pick one of the offered dolls (subset presented per page) ──
    EventOptionSpec {
        option_id: "doll_daughter",
        loc_key: "",
        default_label: "Daughter of the Wind",
        default_description: "Take this doll.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: doll_room_pick_daughter,
    },
    EventOptionSpec {
        option_id: "doll_mr_struggles",
        loc_key: "",
        default_label: "Mr. Struggles",
        default_description: "Take this doll.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: doll_room_pick_mr_struggles,
    },
    EventOptionSpec {
        option_id: "doll_bingbong",
        loc_key: "",
        default_label: "Bing Bong",
        default_description: "Take this doll.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: doll_room_pick_bingbong,
    },
];

/// Port pin: C# `DollRoom.cs`. `is_allowed`: `allowed_act_2_only`
/// (`CurrentActIndex == 1`). Multi-page (bead `sts2-rs-m3n0g.3`): Take Some
/// Time / Examine open a doll-pick sub-page via [`doll_room_generate_options`].
pub const DOLL_ROOM: EventSpec = EventSpec {
    id: EventId::DollRoom,
    python_name: "DollRoom",
    is_allowed: allowed_act_2_only,
    generate_options: doll_room_generate_options,
    options: DOLL_ROOM_OPTIONS,
};

// ── EndlessConveyor ──────────────────────────────────────────────────
//
// C# `EndlessConveyor` (EndlessConveyor.cs, identical in v0.107.1 and
// v0.111.0 apart from decompiler names) is a belt loop:
//
// - `CalculateVars` (event start) runs `RollDish()` once.
// - Page INITIAL offers [grab the current dish | Observe Chef].
// - `GrabSomethingOffTheBelt` pays 40 gold (free for Golden Fysh), awaits the
//   dish's action, rolls the next dish, and shows page GRAB_SOMETHING_OFF_THE_
//   BELT: [grab the new dish | Leave]. The grab option is LOCKED (disabled)
//   below 40 gold.
// - `RollDish` (EndlessConveyor.cs:212-262): `NumOfGrabs++`; every 5th roll
//   is Seapunk Salad with no draw; otherwise the weighted pool [Caviar 6,
//   Spicy Snappy 3, Jelly Liver 3, Fried Eel 3] + Suspicious Condiment 3 (an
//   open potion slot) + Clam Roll 6 (HP below max) + Golden Fysh 1 (after the
//   first roll), minus the last dish, picked with ONE `base.Rng.NextFloat()`
//   times the float weight sum and a strict `<` over float running sums.
//
// Every draw (`RollDish`, Spicy Snappy's and Observe's `NextItem`, Jelly
// Liver's transform) is on the event's own stream, kept in
// `event_state.rng` across pages. `event_state.scratch` holds
// `[NumOfGrabs, last dish, current dish]` (dish codes below, -1 = none).

/// `EndlessConveyor::grab` per-dish cost — C#
/// `EndlessConveyor.cs` `CanonicalVars[0] = GoldVar(40)` (Python
/// pin `act2.py:213`'s 35 drifted from the shipped game; C#
/// `CalculateVars` only re-rolls the current dish, leaves the Gold
/// base value unchanged). Bead `sts2-rs-vfv16`.
pub const ENDLESS_CONVEYOR_GRAB_COST: i32 = 40;
/// `GoldVar("GoldenFyshGold", 75)` (EndlessConveyor.cs:94).
pub const ENDLESS_CONVEYOR_GOLDEN_FYSH_GOLD: i32 = 75;
/// `HealVar("ClamRollHeal", 10m)` (EndlessConveyor.cs:95).
pub const ENDLESS_CONVEYOR_CLAM_ROLL_HEAL: i32 = 10;
/// `MaxHpVar("CaviarMaxHp", 4m)` (EndlessConveyor.cs:96).
pub const ENDLESS_CONVEYOR_CAVIAR_MAX_HP: i32 = 4;

/// `EndlessConveyor` event-RNG stream name (C# `Id.Entry`, UPPER_SNAKE).
/// `EndlessConveyor` does not override `IsShared`, so the single-player player
/// term applies. v0.107.1 additive player term is 0 (`EventModel.cs:238`
/// GetPlayerSlotIndex == 0; v0.103.2 used `Owner.NetId == 1`,
/// `EventModel.cs:193`). Same construction as the `SLIPPERY_BRIDGE` /
/// `WELCOME_TO_WONGOS` streams.
const ENDLESS_CONVEYOR_STREAM: &str = "ENDLESS_CONVEYOR";

/// One dish on the belt. The discriminant is the `event_state.scratch` code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i16)]
enum ConveyorDish {
    Caviar = 0,
    SpicySnappy = 1,
    JellyLiver = 2,
    FriedEel = 3,
    SuspiciousCondiment = 4,
    ClamRoll = 5,
    GoldenFysh = 6,
    SeapunkSalad = 7,
}

/// "No dish" (C# `_lastDishId = ""` / an unset `_currentDish`).
const CONVEYOR_NO_DISH: i16 = -1;

impl ConveyorDish {
    const ALL: [Self; 8] = [
        Self::Caviar,
        Self::SpicySnappy,
        Self::JellyLiver,
        Self::FriedEel,
        Self::SuspiciousCondiment,
        Self::ClamRoll,
        Self::GoldenFysh,
        Self::SeapunkSalad,
    ];

    fn from_code(code: i16) -> Option<Self> {
        Self::ALL.into_iter().find(|d| *d as i16 == code)
    }

    /// The `Dish.weight` float (EndlessConveyor.cs:221-246).
    const fn weight(self) -> f32 {
        match self {
            Self::Caviar | Self::ClamRoll => 6.0,
            Self::SpicySnappy | Self::JellyLiver | Self::FriedEel | Self::SuspiciousCondiment => {
                3.0
            }
            Self::GoldenFysh => 1.0,
            Self::SeapunkSalad => 0.0,
        }
    }

    /// `Dish.optionKey` = `ENDLESS_CONVEYOR.pages.ALL.options.<id>`.
    const fn loc_key(self) -> &'static str {
        match self {
            Self::Caviar => "ENDLESS_CONVEYOR.pages.ALL.options.CAVIAR",
            Self::SpicySnappy => "ENDLESS_CONVEYOR.pages.ALL.options.SPICY_SNAPPY",
            Self::JellyLiver => "ENDLESS_CONVEYOR.pages.ALL.options.JELLY_LIVER",
            Self::FriedEel => "ENDLESS_CONVEYOR.pages.ALL.options.FRIED_EEL",
            Self::SuspiciousCondiment => "ENDLESS_CONVEYOR.pages.ALL.options.SUSPICIOUS_CONDIMENT",
            Self::ClamRoll => "ENDLESS_CONVEYOR.pages.ALL.options.CLAM_ROLL",
            Self::GoldenFysh => "ENDLESS_CONVEYOR.pages.ALL.options.GOLDEN_FYSH",
            Self::SeapunkSalad => "ENDLESS_CONVEYOR.pages.ALL.options.SEAPUNK_SALAD",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Caviar => "Caviar (40g)",
            Self::SpicySnappy => "Spicy Snappy (40g)",
            Self::JellyLiver => "Jelly Liver (40g)",
            Self::FriedEel => "Fried Eel (40g)",
            Self::SuspiciousCondiment => "Suspicious Condiment (40g)",
            Self::ClamRoll => "Clam Roll (40g)",
            Self::GoldenFysh => "Golden Fysh (free)",
            Self::SeapunkSalad => "Seapunk Salad (40g)",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::Caviar => "Gain 4 Max HP.",
            Self::SpicySnappy => "Upgrade a random card.",
            Self::JellyLiver => "Transform a card.",
            Self::FriedEel => "Gain a random Colorless card.",
            Self::SuspiciousCondiment => "Procure a random potion.",
            Self::ClamRoll => "Heal 10 HP.",
            Self::GoldenFysh => "Gain 75 gold.",
            Self::SeapunkSalad => "Add Feeding Frenzy to your deck.",
        }
    }
}

/// The belt's `[NumOfGrabs, _lastDishId, _currentDish]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ConveyorBelt {
    num_grabs: i16,
    last: i16,
    current: i16,
}

/// The weighted candidates of a non-Seapunk `RollDish`, in C# list order
/// (EndlessConveyor.cs:219-238), for a belt whose `NumOfGrabs` is already
/// incremented.
fn conveyor_dish_pool(state: &RunState, belt: &ConveyorBelt) -> Vec<ConveyorDish> {
    use ConveyorDish as D;
    let mut list = vec![D::Caviar, D::SpicySnappy, D::JellyLiver, D::FriedEel];
    // `Owner.HasOpenPotionSlots`.
    if state.potions.len() < state.potion_belt_cap() {
        list.push(D::SuspiciousCondiment);
    }
    if state.current_hp != state.max_hp {
        list.push(D::ClamRoll);
    }
    if belt.num_grabs > 1 {
        list.push(D::GoldenFysh);
    }
    list.retain(|d| *d as i16 != belt.last);
    list
}

/// C# `RollDish` (EndlessConveyor.cs:212-262) against the player's state
/// right now (open potion slot, HP below max).
fn conveyor_roll_dish(
    state: &RunState,
    belt: &mut ConveyorBelt,
    rng: &mut sts2_core::dotnet_rng::Rng,
) {
    use ConveyorDish as D;
    belt.num_grabs += 1;
    if belt.num_grabs % 5 == 0 {
        belt.last = D::SeapunkSalad as i16;
        belt.current = D::SeapunkSalad as i16;
        return;
    }
    let list = conveyor_dish_pool(state, belt);
    // `float num3 += weight; float num4 = NextFloat() * num3;` then
    // `num5 += weight; if (num4 < num5)` — all float.
    let total = list.iter().fold(0.0_f32, |acc, d| acc + d.weight());
    let roll = rng.next_float_f32() * total;
    let mut running = 0.0_f32;
    for dish in list {
        running += dish.weight();
        if roll < running {
            belt.last = dish as i16;
            belt.current = dish as i16;
            return;
        }
    }
    // A `NextFloat()` that rounds up to 1.0f matches nothing: C# keeps the
    // previous dish (no assignment in the loop).
}

/// The belt after event start: the stored scratch, or C#'s `CalculateVars`
/// roll on a fresh event stream when nothing has been stored yet. Returns
/// the event stream positioned after every draw so far.
fn conveyor_belt_and_rng(state: &RunState) -> (ConveyorBelt, sts2_core::dotnet_rng::Rng) {
    if let [num_grabs, last, current] = state.event_state.scratch[..] {
        let rng = state
            .event_state
            .rng
            .clone()
            .unwrap_or_else(|| state.rng.named(ENDLESS_CONVEYOR_STREAM));
        return (
            ConveyorBelt {
                num_grabs,
                last,
                current,
            },
            rng,
        );
    }
    let mut belt = ConveyorBelt {
        num_grabs: 0,
        last: CONVEYOR_NO_DISH,
        current: CONVEYOR_NO_DISH,
    };
    let mut rng = state.rng.named(ENDLESS_CONVEYOR_STREAM);
    conveyor_roll_dish(state, &mut belt, &mut rng);
    (belt, rng)
}

fn conveyor_store(state: &mut RunState, belt: ConveyorBelt, rng: sts2_core::dotnet_rng::Rng) {
    state.event_state.scratch = vec![belt.num_grabs, belt.last, belt.current];
    state.event_state.rng = Some(rng);
}

/// C# `CalculateVars` → `RollDish()` at event start, on the state the
/// player enters with (before any out-of-combat potion use could open a
/// slot). Idempotent per event visit.
pub fn endless_conveyor_before_event_started(state: &mut RunState) {
    if state.event_state.scratch.is_empty() {
        let (belt, rng) = conveyor_belt_and_rng(state);
        conveyor_store(state, belt, rng);
    }
}

/// C# `GrabSomethingOffTheBelt` (EndlessConveyor.cs:121-135).
fn endless_conveyor_grab(state: &mut RunState) -> EventOutcome {
    use ConveyorDish as D;
    let (mut belt, mut rng) = conveyor_belt_and_rng(state);
    let dish = D::from_code(belt.current);
    if dish != Some(D::GoldenFysh) {
        state.gold = state.gold.saturating_sub(ENDLESS_CONVEYOR_GRAB_COST).max(0);
    }
    let description = match dish {
        Some(D::Caviar) => {
            state.gain_max_hp(ENDLESS_CONVEYOR_CAVIAR_MAX_HP);
            "Ate Caviar: +4 Max HP."
        }
        Some(D::ClamRoll) => {
            state.heal(ENDLESS_CONVEYOR_CLAM_ROLL_HEAL);
            "Ate a Clam Roll: healed."
        }
        Some(D::GoldenFysh) => {
            state.gain_gold(ENDLESS_CONVEYOR_GOLDEN_FYSH_GOLD);
            "Caught the Golden Fysh: +75 gold."
        }
        Some(D::SpicySnappy) => {
            // `base.Rng.NextItem(deck.Where(IsUpgradable))` → Upgrade; an
            // empty list draws nothing (EndlessConveyor.cs:197-206).
            let upgradable: Vec<usize> = state
                .master_deck
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.is_upgradable().then_some(i))
                .collect();
            if !upgradable.is_empty() {
                let pick = rng.next_int_exclusive(0, upgradable.len() as i32) as usize;
                state.master_deck[upgradable[pick]].upgrade_for(state.game_version);
            }
            "Ate a Spicy Snappy: upgraded a card."
        }
        Some(D::JellyLiver) => {
            conveyor_jelly_liver(state, &mut rng);
            "Ate a Jelly Liver: transform a card."
        }
        Some(D::FriedEel) => {
            // `CardFactory.CreateForReward(owner, 1,
            // ForNonCombatWithDefaultOdds(ColorlessCardPool))` → deck
            // (EndlessConveyor.cs:186-191): base odds, no pity, no upgrade
            // roll, on PlayerRng.Rewards; the reward-option hooks run.
            let ids = crate::cards::spec_index::roll_colorless_reward_options_no_upgrade(state, 1);
            if let Some(card) = ids
                .first()
                .and_then(|&id| crate::cards::spec_index::card_pod_from_id(id, false))
            {
                super::shared::add_created_for_reward_card(state, card);
            }
            "Ate a Fried Eel: gained a colorless card."
        }
        Some(D::SuspiciousCondiment) => {
            // One `PlayerRng.Rewards.NextItem` over character ++ shared
            // potions, offered as a populated PotionReward and awaited
            // (EndlessConveyor.cs:150-161). Claimed here, before the next
            // `RollDish` reads `HasOpenPotionSlots` — the drain would claim
            // it only after this handler returns. The dish is only on the
            // belt with a slot open, so the claim always fits.
            if let Some(kind) = crate::potions::all::roll_event_potion_full_pool(
                &mut state.rng.rewards,
                state.character_id,
            ) {
                let _ = state.claim_potion_reward(kind);
            }
            "Ate a Suspicious Condiment: gained a potion."
        }
        Some(D::SeapunkSalad) => {
            // `CreateCard<FeedingFrenzy>` → `CardPileCmd.Add(Deck)`.
            if let Some(card) = crate::cards::spec_index::card_pod_from_id(
                sts2_core::cards::CardId::FeedingFrenzyCard,
                false,
            ) {
                state.add_card_to_deck_with_hooks(
                    card,
                    sts2_core::relic_effects::AddSource::EventGain,
                );
            }
            "Ate a Seapunk Salad: gained Feeding Frenzy."
        }
        None => "The belt was empty.",
    };
    conveyor_roll_dish(state, &mut belt, &mut rng);
    conveyor_store(state, belt, rng);
    state.event_state.page = 1;
    EventOutcome {
        finished: false,
        description: description.to_string(),
        effects: Vec::new(),
    }
}

/// C# `JellyLiver` (EndlessConveyor.cs:163-170): pick 1 deck card
/// (`FromDeckForTransformation`), then `TransformToRandom(card, base.Rng)`.
/// The transform's one event-stream draw precedes the next `RollDish`, so
/// it is reserved here: an interactive host queues the pick with the
/// stream position (`skip_reads`) and this stream skips past it; a
/// non-interactive host transforms the first deck card now (the
/// MorphicGrove convention).
fn conveyor_jelly_liver(state: &mut RunState, rng: &mut sts2_core::dotnet_rng::Rng) {
    let Some(original) = state.master_deck.first().map(|c| c.card_id) else {
        return;
    };
    let skip_reads = rng.counter();
    if state.interactive_card_transform {
        state
            .pending_deck_transforms
            .push(sts2_core::PendingDeckTransform {
                event_stream_roll: Some((ENDLESS_CONVEYOR_STREAM, skip_reads)),
                ..Default::default()
            });
        let _ = rng.next_int(0, 1);
        return;
    }
    if let Some(new_id) = crate::cards::spec_index::roll_transform_replacement_event_stream(
        state,
        state.character_id,
        original,
        ENDLESS_CONVEYOR_STREAM,
        skip_reads,
    ) {
        let _ = rng.next_int(0, 1);
        if let Some(card) = crate::cards::spec_index::card_pod_from_id(new_id, false) {
            state.master_deck.remove(0);
            state.add_card_to_deck_with_hooks(card, sts2_core::relic_effects::AddSource::EventGain);
        }
    }
}

fn endless_conveyor_observe(state: &mut RunState) -> EventOutcome {
    // C# `EndlessConveyor.ObserveChef` (EndlessConveyor.cs:268-278):
    //   CardCmd.Upgrade(base.Rng.NextItem(Deck.Cards.Where(IsUpgradable)))
    // on the event stream, after `CalculateVars`' `RollDish` draw.
    let (_, mut rng) = conveyor_belt_and_rng(state);
    let upgradable: Vec<usize> = state
        .master_deck
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.is_upgradable().then_some(i))
        .collect();
    if upgradable.is_empty() {
        return EventOutcome::finished("Observed chef — no upgradable card.");
    }
    let pick = rng.next_int_exclusive(0, upgradable.len() as i32) as usize;
    state.master_deck[upgradable[pick]].upgrade_for(state.game_version);
    EventOutcome::finished("Observed chef — upgraded a card.")
}

fn endless_conveyor_leave(_state: &mut RunState) -> EventOutcome {
    EventOutcome::finished("Left the conveyor.")
}

/// Page INITIAL = [grab, observe]; page GRAB_SOMETHING_OFF_THE_BELT =
/// [grab, leave]. The grab row names the current dish, or is the disabled
/// LOCKED row below 40 gold (`GenerateGrabSomethingOffTheBeltOption`,
/// EndlessConveyor.cs:137-144 — Golden Fysh included).
fn endless_conveyor_generate_options(state: &RunState, spec: &EventSpec) -> Vec<EventOptionInfo> {
    use sts2_core::events::EventVar;
    let (belt, _) = conveyor_belt_and_rng(state);
    let base = super::default_option_info_from_spec(state, spec);
    let take = |id: &str| {
        base.iter()
            .find(|o| o.option_id == id)
            .cloned()
            .expect("conveyor option")
    };
    let mut grab = take("grab");
    match ConveyorDish::from_code(belt.current) {
        Some(dish) if state.gold >= ENDLESS_CONVEYOR_GRAB_COST => {
            grab.label = dish.label().to_string();
            grab.description = dish.description().to_string();
            grab.loc_key = dish.loc_key();
            grab.vars = vec![
                ("Gold", EventVar::Int(i64::from(ENDLESS_CONVEYOR_GRAB_COST))),
                (
                    "GoldenFyshGold",
                    EventVar::Int(i64::from(ENDLESS_CONVEYOR_GOLDEN_FYSH_GOLD)),
                ),
                (
                    "ClamRollHeal",
                    EventVar::Int(i64::from(ENDLESS_CONVEYOR_CLAM_ROLL_HEAL)),
                ),
                (
                    "CaviarMaxHp",
                    EventVar::Int(i64::from(ENDLESS_CONVEYOR_CAVIAR_MAX_HP)),
                ),
            ];
        }
        _ => {
            grab.enabled = false;
            grab.label = "Broke".to_string();
            grab.description = "Not enough gold.".to_string();
            grab.loc_key = "ENDLESS_CONVEYOR.pages.ALL.options.LOCKED";
        }
    }
    let second = if state.event_state.page == 0 {
        take("observe")
    } else {
        take("leave")
    };
    vec![grab, second]
}

const ENDLESS_CONVEYOR_OPTIONS: &[EventOptionSpec] = &[
    EventOptionSpec {
        option_id: "grab",
        loc_key: "",
        default_label: "Grab Something (40g)",
        default_description: "Take the current dish (40g per grab).",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: endless_conveyor_grab,
    },
    EventOptionSpec {
        option_id: "observe",
        loc_key: "ENDLESS_CONVEYOR.pages.INITIAL.options.OBSERVE_CHEF",
        default_label: "Observe Chef",
        default_description: "Upgrade 1 random card (free).",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: endless_conveyor_observe,
    },
    EventOptionSpec {
        option_id: "leave",
        loc_key: "ENDLESS_CONVEYOR.pages.GRAB_SOMETHING_OFF_THE_BELT.options.LEAVE",
        default_label: "Leave",
        default_description: "Leave the conveyor.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: endless_conveyor_leave,
    },
];

/// `is_allowed`: `endless_conveyor_allowed` (`Gold >= 120`). Multi-page belt
/// loop; see the section comment above.
pub const ENDLESS_CONVEYOR: EventSpec = EventSpec {
    id: EventId::EndlessConveyor,
    python_name: "EndlessConveyor",
    is_allowed: endless_conveyor_allowed,
    generate_options: endless_conveyor_generate_options,
    options: ENDLESS_CONVEYOR_OPTIONS,
};

// ── FakeMerchant ─────────────────────────────────────────────────────

/// Every fake relic's `MerchantCost` (`FakeAnchor.cs:15` and its eight
/// siblings). The price actually charged is this jittered once per stocked
/// slot at event start — see [`fake_merchant_before_event_started`].
pub const FAKE_MERCHANT_BUY_COST: i32 = 50;

/// `Id.Entry` for FakeMerchant's own RNG stream, used to shuffle the merchant
/// inventory in `BeforeEventStarted`. FakeMerchant is a SHARED event
/// (`IsShared => true`), so the C# seed folds in NO NetId offset:
/// `base.Rng = new Rng(RunState.Rng.Seed + 0 + hash("FAKE_MERCHANT"))`
/// (`EventModel.BeginEvent`, `IsShared ? 0 : NetId`) — same convention as
/// `JungleMazeAdventure`.
const FAKE_MERCHANT_STREAM: &str = "FAKE_MERCHANT";

/// C# `FakeMerchant._inventoryRelics` — the 9 fake relics, in source order
/// (`FakeMerchant.cs:23-33`). `BeforeEventStarted` shuffles this list with
/// the event's own RNG and stocks the first 6.
const FAKE_MERCHANT_INVENTORY: [RelicId; 9] = [
    RelicId::FakeAnchor,
    RelicId::FakeBloodVial,
    RelicId::FakeHappyFlower,
    RelicId::FakeLeesWaffle,
    RelicId::FakeMango,
    RelicId::FakeOrichalcum,
    RelicId::FakeSneckoEye,
    RelicId::FakeStrikeDummy,
    RelicId::FakeVenerableTeaSet,
];

/// Number of fake relics the merchant stocks (C# `.Take(6)`).
const FAKE_MERCHANT_STOCK_SIZE: usize = 6;

/// Roll the merchant's stocked inventory: shuffle [`FAKE_MERCHANT_INVENTORY`]
/// with the event's own RNG and take the first 6. Mirrors C#
/// `FakeMerchant.BeforeEventStarted`:
/// `_inventoryRelics.ToList().UnstableShuffle(base.Rng).Take(6)`. The event
/// RNG is non-shared (re-derived from the run seed + `hash("FAKE_MERCHANT")`),
/// so this consumes NO shared run RNG and is deterministic — re-derivable per
/// option handler, which is all the option-model needs (each FakeMerchant
/// visit resolves exactly one option: buy XOR throw-foul XOR leave).
fn fake_merchant_stock(state: &RunState) -> Vec<RelicId> {
    let mut idx: Vec<usize> = (0..FAKE_MERCHANT_INVENTORY.len()).collect();
    state.rng.named(FAKE_MERCHANT_STREAM).shuffle(&mut idx);
    idx.truncate(FAKE_MERCHANT_STOCK_SIZE);
    idx.into_iter()
        .map(|i| FAKE_MERCHANT_INVENTORY[i])
        .collect()
}

/// C# `FakeMerchant.BeforeEventStarted` side effect on the SHARED run RNG:
/// each of the 6 stocked `new MerchantRelicEntry(relic, Owner)` runs
/// `SetModel → CalcCost`, i.e. `Math.Round(MerchantCost *
/// PlayerRng.Shops.NextFloat(0.85f, 1.15f))` (`MerchantRelicEntry.cs:45`,
/// `FakeMerchant.cs:116`; identical in v0.111.0). That is 6 `Shops` draws
/// on event START, whatever option is then taken, so skipping them shifts
/// every later shop (act23 probe: Ironclad/Regent 1009, Ironclad/Defect
/// 1015, Necrobinder 1004). The rolled prices, in stock order, go to
/// `event_state.scratch` for the buy option. Idempotent per event visit.
pub fn fake_merchant_before_event_started(state: &mut RunState) {
    if !state.event_state.scratch.is_empty() {
        return;
    }
    for _ in 0..FAKE_MERCHANT_STOCK_SIZE {
        // `(float)MerchantCost * NextFloat(0.85f, 1.15f)`, banker's — the
        // single-precision product of `MerchantRelicEntry.CalcCost`.
        let jitter = sts2_core::shop::merchant_relic_jitter(&mut state.rng.shops);
        let price = i16::try_from(sts2_core::shop::merchant_round(
            FAKE_MERCHANT_BUY_COST,
            jitter,
        ))
        .unwrap_or(i16::MAX);
        state.event_state.scratch.push(price);
    }
}

/// Price of the first stocked fake relic: its event-start roll, or the bare
/// `MerchantCost` when the event was bound without
/// [`fake_merchant_before_event_started`] (test scaffolding).
fn fake_merchant_first_price(state: &RunState) -> i32 {
    state
        .event_state
        .scratch
        .first()
        .map_or(FAKE_MERCHANT_BUY_COST, |&p| i32::from(p))
}

/// `Buy a Relic`: purchase one stocked fake relic at its rolled price. The
/// custom C# merchant UI lets the player pick any of the 6 displayed; the
/// option-model purchases the first stocked (shuffle-order) entry. Mirrors
/// the `MerchantRelicEntry` buy → `LoseGold(Cost)` → `RelicCmd.Obtain`
/// (no `ModifyMerchantPrice` — the event room is not a `MerchantRoom`,
/// `MerchantEntry.cs:24`).
fn fake_merchant_buy(state: &mut RunState) -> EventOutcome {
    let price = fake_merchant_first_price(state);
    state.gold = state.gold.saturating_sub(price);
    let stock = fake_merchant_stock(state);
    let relic = stock.first().copied().unwrap_or(RelicId::FakeAnchor);
    state.grant_event_relic(relic);
    EventOutcome::finished(format!("Bought a fake relic ({price}g)."))
}

/// `Throw Foul Potion`: C# `FakeMerchant.FoulPotionThrown` →
/// `EnterCombatWithoutExitingEvent<FakeMerchantEventEncounter>(relicRewards,
/// shouldResumeAfterCombat: false)`.
///
/// Bead `sts2-rs-ck9z6` (GH #166): enters a real `Phase::Combat` against the
/// FakeMerchantMonster. On a clean victory the combat-room extra rewards are
/// granted — `FakeMerchantsRug` PLUS every *stocked* fake inventory relic, in
/// the inventory's shuffle order, exactly as C# `FoulPotionThrown` builds its
/// `List<RelicReward>` (`new RelicReward<FakeMerchantsRug>` then one per
/// `Inventory.RelicEntries.Where(IsStocked)`), after the fight's own
/// Monster-room rewards: the encounter's flat 300 gold
/// (`FakeMerchantEventEncounter.MinGoldReward/MaxGoldReward`), the potion
/// odds and a card reward (GH #817). The thrown FoulPotion is
/// consumed (the `Custom` merchant UI throws the potion the player holds).
///
/// In the option-model a FakeMerchant visit resolves exactly one option, so
/// throw-foul sees the full 6-relic stock (no prior buy in the same
/// resolution) — matching a fresh C# `FoulPotionThrown` where nothing was
/// purchased.
fn fake_merchant_throw_foul(state: &mut RunState) -> EventOutcome {
    // Consume the thrown FoulPotion if the player holds one (C# throws the
    // held potion; the event is gated on having a FoulPotion or 100 gold).
    if let Some(pos) = state
        .potions
        .iter()
        .position(|&p| p == sts2_core::potions::PotionKind::FoulPotion)
    {
        let _ = state.remove_potion_from_belt(pos);
    }
    let mut victory_rewards = vec![Reward::Relic(RelicReward::new(RelicId::FakeMerchantsRug))];
    victory_rewards.extend(
        fake_merchant_stock(state)
            .into_iter()
            .map(|id| Reward::Relic(RelicReward::new(id))),
    );
    state.event_combat = Some(EventCombatRequest::no_resume(
        EventEncounter::FakeMerchant,
        EventId::FakeMerchant,
        victory_rewards,
    ));
    EventOutcome {
        finished: false,
        description: "You hurl the foul potion. The merchant drops the act — fight!".to_string(),
        effects: Vec::new(),
    }
}

/// Present `buy, throw_foul, leave` in that fixed order, disabling what the
/// game would not let the player do (GH #885). The fight only starts from
/// a thrown Foul Potion (`FoulPotion.OnUse` → `FakeMerchant.FoulPotionThrown`),
/// and the merchant UI only sells a stocked relic the player can afford. A
/// disabled option keeps its slot: the live bridge and the policy index by
/// position.
fn fake_merchant_generate_options(state: &RunState, spec: &EventSpec) -> Vec<EventOptionInfo> {
    let can_buy =
        state.gold >= fake_merchant_first_price(state) && !fake_merchant_stock(state).is_empty();
    let has_foul = state
        .potions
        .contains(&sts2_core::potions::PotionKind::FoulPotion);
    let mut options = super::default_option_info_from_spec(state, spec);
    for opt in &mut options {
        opt.enabled = match opt.option_id {
            "buy" => can_buy,
            "throw_foul" => has_foul,
            _ => true,
        };
    }
    options
}

fn fake_merchant_leave(_state: &mut RunState) -> EventOutcome {
    EventOutcome::finished("Left the fake merchant.")
}

const FAKE_MERCHANT_OPTIONS: &[EventOptionSpec] = &[
    EventOptionSpec {
        option_id: "buy",
        loc_key: "",
        default_label: "Buy a Relic (50g)",
        default_description: "Purchase a fake relic.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: fake_merchant_buy,
    },
    EventOptionSpec {
        option_id: "throw_foul",
        loc_key: "",
        default_label: "Throw Foul Potion",
        default_description: "Fight for real rewards.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: fake_merchant_throw_foul,
    },
    EventOptionSpec {
        option_id: "leave",
        loc_key: "",
        default_label: "Leave",
        default_description: "Leave the merchant.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: fake_merchant_leave,
    },
];

/// `is_allowed`: `fake_merchant_allowed`
/// (`current_act_index >= 1 && (gold >= 100 || has_foul)`) — wired.
/// `buy` needs 50 gold and `throw_foul` a held Foul Potion (GH #885).
pub const FAKE_MERCHANT: EventSpec = EventSpec {
    id: EventId::FakeMerchant,
    python_name: "FakeMerchant",
    is_allowed: fake_merchant_allowed,
    generate_options: fake_merchant_generate_options,
    options: FAKE_MERCHANT_OPTIONS,
};

// ── FieldOfManSizedHoles ─────────────────────────────────────────────

fn field_of_man_sized_holes_resist(state: &mut RunState) -> EventOutcome {
    // C# `FieldOfManSizedHoles.Resist`: `FromDeckForRemoval(CardsVar=2)` →
    // remove, THEN `AddCursesToDeck<Normality>`. Normality lands after the
    // removal, so defer it past the batch (interactive) / append after the auto
    // front-removal — same shape as `wellspring_bathe` / `luminous_choir_reach`.
    use sts2_core::cards::CardId;
    let normality = crate::cards::spec_index::card_pod_from_id(CardId::Normality, false);
    if state.interactive_card_removal {
        state.pending_deck_removals += 2;
        if let Some(card) = normality {
            state.pending_post_removal_cards.push(card);
        }
    } else {
        let _ = state.remove_cards_from_deck(2);
        if let Some(card) = normality {
            state.add_card_to_deck_with_hooks(card, sts2_core::relic_effects::AddSource::EventGain);
        }
    }
    EventOutcome::finished("Resisted — removed 2 cards and gained a Normality curse.")
}

/// Base `EnchantmentModel.CanEnchant` (PerfectFit has no override): not
/// Status/Curse, unenchanted, concrete cost (`cost >= 0` folds Unplayable /
/// X-cost). Mirrors [`super::any_enchantable_deck_card`]'s per-card filter.
fn field_of_man_can_enchant(c: &sts2_core::cards::Card) -> bool {
    use sts2_core::cards::CardType;
    !matches!(c.card_type, CardType::Status | CardType::Curse)
        && !c.is_quest()
        && c.enchantments.is_empty()
        && c.base_cost() >= 0
}

fn field_of_man_sized_holes_enter(state: &mut RunState) -> EventOutcome {
    // C# `FieldOfManSizedHoles.EnterYourHole`: `FromDeckForEnchantment(PerfectFit,
    // 1)` pick-1 → `Enchant<PerfectFit>(card, 1)`. Interactive host queues a
    // `pending_deck_enchants`; the grid auto-takes at <=1 eligible — same shape
    // as `sapphire_seed_plant`.
    use sts2_core::cards::EnchantmentId;
    let eligible = state
        .master_deck
        .iter()
        .filter(|c| field_of_man_can_enchant(c))
        .count();
    if state.interactive_card_transform && eligible > 1 {
        state
            .pending_deck_enchants
            .push(sts2_core::PendingDeckEnchant {
                enchantment: EnchantmentId::PerfectFit,
                amount: 1,
                type_restriction: None,
                secondary_type: None,
                optional: false,
            });
        EventOutcome::finished("Pick a card to enchant with Perfect Fit.")
    } else {
        if let Some(card) = state
            .master_deck
            .iter_mut()
            .find(|c| field_of_man_can_enchant(c))
        {
            card.enchantments.push((EnchantmentId::PerfectFit, 1));
        }
        EventOutcome::finished("Enchanted a card with Perfect Fit.")
    }
}

const FIELD_OF_MAN_SIZED_HOLES_OPTIONS: &[EventOptionSpec] = &[
    EventOptionSpec {
        option_id: "resist",
        loc_key: "FIELD_OF_MAN_SIZED_HOLES.pages.INITIAL.options.RESIST",
        default_label: "Resist",
        default_description: "Remove 2 cards, gain Normality curse.",
        default_vars: &[
            ("Cards", EventVarSpec::Int(2)),
            ("ResistCurse", EventVarSpec::Text("Normality")),
        ],
        vars_fn: None,
        title_vars_fn: None,
        handler: field_of_man_sized_holes_resist,
    },
    EventOptionSpec {
        option_id: "enter",
        loc_key: "FIELD_OF_MAN_SIZED_HOLES.pages.INITIAL.options.ENTER_YOUR_HOLE",
        default_label: "Enter Your Hole",
        default_description: "Enchant a card with Perfect Fit.",
        default_vars: &[("Enchantment", EventVarSpec::Text("Perfect Fit"))],
        vars_fn: None,
        title_vars_fn: None,
        handler: field_of_man_sized_holes_enter,
    },
];

/// `is_allowed`: `any_enchantable_deck_card` (`any(can_enchant_card)`) — wired.
/// resist (remove 2 + Normality) / enter (PerfectFit enchant) wired via the
/// deck-mutation surface (`sts2-rs-m3n0g.2`).
pub const FIELD_OF_MAN_SIZED_HOLES: EventSpec = EventSpec {
    id: EventId::FieldOfManSizedHoles,
    python_name: "FieldOfManSizedHoles",
    is_allowed: any_enchantable_deck_card,
    generate_options: super::default_option_info_from_spec,
    options: FIELD_OF_MAN_SIZED_HOLES_OPTIONS,
};

// ── JungleMazeAdventure ──────────────────────────────────────────────

/// `JungleMazeAdventure` `SoloHp` cost. C# DynamicVar `BaseValue = 18`
/// (Unblockable | Unpowered). Matches across the port.
pub const JUNGLE_MAZE_SOLO_DAMAGE: i32 = 18;
/// `SoloGold` base before the per-event RNG swing (C# DynamicVar
/// `BaseValue = 150m`).
const JUNGLE_MAZE_SOLO_GOLD_BASE: i128 = 150;
/// `JoinForcesGold` base before the per-event RNG swing (C# DynamicVar
/// `BaseValue = 50m`).
const JUNGLE_MAZE_JOIN_GOLD_BASE: i128 = 50;

/// Roll `(solo_gold, join_forces_gold)` off the per-event RNG, faithful
/// to C# `JungleMazeAdventure.CalculateVars` (`JungleMazeAdventure.cs`)
/// plus the per-event stream `EventModel.BeginEvent` constructs:
///
/// ```text
/// Rng = new Rng(RunState.Rng.Seed + hash("JUNGLE_MAZE_ADVENTURE"))  // IsShared ⇒ no NetId
/// SoloGold       = 150 + (float)NextFloat(-15f, 15f)   // CalculateVars draw #1
/// JoinForcesGold =  50 + (float)NextFloat(-15f, 15f)   // CalculateVars draw #2
/// ```
///
/// Both swings come off ONE fresh per-event stream in that exact draw
/// order. The event seeds its own `Rng` (so the run-level streams are
/// untouched), which makes rolling here on demand equivalent to C#'s
/// one-shot `CalculateVars` — `Rng::with_name(seed, name)` reproduces
/// `new Rng(seed + GetDeterministicHashCode(name))`. The `(float)` cast
/// in `Rng.NextFloat(min,max) = (float)(NextDouble()*(max-min)+min)` is
/// replicated with `as f32`, and the int gold delta truncates toward
/// zero exactly like C# `PlayerCmd.GainGold`'s `player.Gold += (int)amount`.
///
/// Supersedes the deterministic-midpoint pin (`SoloGold = 150`,
/// `JoinForcesGold = 50`, deferred to `6ynt.4.rng`): the Python pin's
/// `up_front.next_int(-15, 16)` was both the wrong stream and a discrete
/// int — C# rolls a float on the event's own stream. Surfaced by the
/// run-RNG lockstep (seed 1: C# +148 vs Rust +150).
fn jungle_maze_gold(rs: &RunState) -> (i32, i32) {
    let ((solo_n, solo_d), (join_n, join_d)) = jungle_maze_gold_exact(rs);
    // Add-then-truncate matches `(int)(150m + decimal_swing)` (the swing can
    // be negative, so the truncation must follow the addition).
    (
        i32::try_from(solo_n / solo_d).unwrap_or(i32::MAX),
        i32::try_from(join_n / join_d).unwrap_or(i32::MAX),
    )
}

/// The exact C# `decimal` `(SoloGold, JoinForcesGold)` as `numer / denom`
/// pairs. `PlayerCmd.GainGold` applies Bowler Hat to this decimal and
/// truncates once, so the grant must not go through the truncated ints of
/// [`jungle_maze_gold`].
fn jungle_maze_gold_exact(rs: &RunState) -> ((i128, i128), (i128, i128)) {
    let mut rng = rs.rng.named("JUNGLE_MAZE_ADVENTURE");
    // `as f32` mirrors C#'s `(float)` NextFloat result; `(decimal)` of that
    // float keeps 7 significant digits.
    let solo = csharp_decimal_from_f32(rng.next_float_range(-15.0, 15.0) as f32);
    let join = csharp_decimal_from_f32(rng.next_float_range(-15.0, 15.0) as f32);
    let add = |base: i128, (n, d): (i128, i128)| (base * d + n, d);
    (
        add(JUNGLE_MAZE_SOLO_GOLD_BASE, solo),
        add(JUNGLE_MAZE_JOIN_GOLD_BASE, join),
    )
}

/// C# `(decimal)someFloat` as an exact `numer / denom`: the conversion
/// rounds to 7 significant digits (`System.Decimal(float)`).
fn csharp_decimal_from_f32(x: f32) -> (i128, i128) {
    let s = format!("{x:.6e}");
    let (mant, exp) = s.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let digits: i128 = mant.replace('.', "").parse().expect("7 signed digits");
    let shift = exp - 6;
    if shift >= 0 {
        (digits * 10i128.pow(shift.unsigned_abs()), 1)
    } else {
        (digits, 10i128.pow(shift.unsigned_abs()))
    }
}

fn jungle_maze_generate_options(state: &RunState, _spec: &EventSpec) -> Vec<EventOptionInfo> {
    let (solo_gold, join_gold) = jungle_maze_gold(state);
    vec![
        EventOptionInfo {
            option_id: "solo",
            loc_key: "JUNGLE_MAZE_ADVENTURE.pages.INITIAL.options.SOLO_QUEST",
            vars: Vec::new(),
            title_vars: Vec::new(),
            label: "Solo Quest".to_string(),
            description: format!("Take {JUNGLE_MAZE_SOLO_DAMAGE} damage, gain {solo_gold} gold."),
            enabled: true,
        },
        EventOptionInfo {
            option_id: "join",
            loc_key: "JUNGLE_MAZE_ADVENTURE.pages.INITIAL.options.JOIN_FORCES",
            vars: Vec::new(),
            title_vars: Vec::new(),
            label: "Join Forces".to_string(),
            description: format!("Gain {join_gold} gold."),
            enabled: true,
        },
    ]
}

fn jungle_maze_solo(state: &mut RunState) -> EventOutcome {
    let (solo_gold, _join) = jungle_maze_gold(state);
    let ((numer, denom), _join) = jungle_maze_gold_exact(state);
    let lost = state.lose_hp(JUNGLE_MAZE_SOLO_DAMAGE);
    state.gain_gold_decimal(numer, denom);
    EventOutcome::finished(format!("Took {lost} damage, gained {solo_gold} gold."))
}

fn jungle_maze_join(state: &mut RunState) -> EventOutcome {
    let (_solo, join_gold) = jungle_maze_gold(state);
    let (_solo, (numer, denom)) = jungle_maze_gold_exact(state);
    state.gain_gold_decimal(numer, denom);
    EventOutcome::finished(format!("Gained {join_gold} gold."))
}

const JUNGLE_MAZE_OPTIONS: &[EventOptionSpec] = &[
    EventOptionSpec {
        option_id: "solo",
        loc_key: "JUNGLE_MAZE_ADVENTURE.pages.INITIAL.options.SOLO_QUEST",
        default_label: "Solo Quest",
        default_description: "Take 18 damage, gain ~150 gold.",
        default_vars: &[("SoloHp", EventVarSpec::Int(JUNGLE_MAZE_SOLO_DAMAGE as i64))],
        vars_fn: Some(|s| vec![("SoloGold", jungle_maze_gold(s).0.into())]),
        title_vars_fn: None,
        handler: jungle_maze_solo,
    },
    EventOptionSpec {
        option_id: "join",
        loc_key: "JUNGLE_MAZE_ADVENTURE.pages.INITIAL.options.JOIN_FORCES",
        default_label: "Join Forces",
        default_description: "Gain ~50 gold.",
        default_vars: &[],
        vars_fn: Some(|s| vec![("JoinForcesGold", jungle_maze_gold(s).1.into())]),
        title_vars_fn: None,
        handler: jungle_maze_join,
    },
];

pub const JUNGLE_MAZE_ADVENTURE: EventSpec = EventSpec {
    id: EventId::JungleMazeAdventure,
    python_name: "JungleMazeAdventure",
    is_allowed: always_allowed,
    generate_options: jungle_maze_generate_options,
    options: JUNGLE_MAZE_OPTIONS,
};

// ── LuminousChoir ────────────────────────────────────────────────────

/// `LuminousChoir` base tribute cost — C# `CanonicalVars` `GoldVar(149)`.
pub const LUMINOUS_CHOIR_TRIBUTE_BASE_COST: i32 = 149;
/// `Id.Entry` for LuminousChoir's own RNG stream. NON-shared event, so the
/// C# event-Rng seed is `RunState.Rng.Seed + NetId + hash(Entry)`; mirror via
/// `Rng::with_name(seed + NetId, Entry)` (same convention as SunkenStatue).
pub const LUMINOUS_CHOIR_STREAM: &str = "LUMINOUS_CHOIR";
/// Fallback when the relic grab bag is exhausted — C# `RelicFactory.FallbackRelic`.
pub const LUMINOUS_CHOIR_FALLBACK_RELIC: RelicId = RelicId::Circlet;

/// C# `LuminousChoir.CalculateVars`: `Gold.BaseValue -= base.Rng.NextInt(0, 50)`
/// → cost `[100, 149]`. Read #1 of the event's private stream; re-derived per
/// call (private stream, so re-deriving preserves in-stream order — mirrors
/// `sunken_statue_rolled_gold`). The old port pinned the midpoint 125, desyncing
/// the gold total vs C# (RP89 Defect).
///
/// C# `Rng.NextInt(int min, int max)` is max-EXCLUSIVE — `NextInt(0, 50)` rolls
/// `[0, 49]`. Use `next_int_exclusive` (Rust's plain `next_int` is inclusive on
/// BOTH ends, so `next_int(0, 50)` would roll `[0, 50]` — one wider, always
/// rolling ≥ C# and so charging 1 LESS gold; the off-by-one gold-class bug,
/// task #14). Bead `sts2-rs-RP14`.
fn luminous_choir_rolled_cost(state: &RunState) -> i32 {
    // v0.107.1 single-player event seed: additive player term is 0
    // (`EventModel.cs:238` GetPlayerSlotIndex == 0; v0.103.2 used NetId == 1).
    let mut rng = state.rng.named(LUMINOUS_CHOIR_STREAM);
    LUMINOUS_CHOIR_TRIBUTE_BASE_COST - rng.next_int_exclusive(0, 50)
}

fn luminous_choir_generate_options(state: &RunState, _spec: &EventSpec) -> Vec<EventOptionInfo> {
    let cost = luminous_choir_rolled_cost(state);
    vec![
        EventOptionInfo {
            option_id: "reach",
            loc_key: "LUMINOUS_CHOIR.pages.INITIAL.options.REACH_INTO_THE_FLESH",
            vars: Vec::new(),
            title_vars: Vec::new(),
            label: "Reach Into the Flesh".to_string(),
            description: "Remove 2 cards, gain Spore Mind curse.".to_string(),
            enabled: true,
        },
        EventOptionInfo {
            option_id: "tribute",
            loc_key: "LUMINOUS_CHOIR.pages.INITIAL.options.OFFER_TRIBUTE",
            vars: Vec::new(),
            title_vars: Vec::new(),
            label: if state.gold >= cost {
                format!("Offer Tribute ({cost}g)")
            } else {
                "Offer Tribute".to_string()
            },
            description: "Gain a relic.".to_string(),
            enabled: state.gold >= cost,
        },
    ]
}

fn luminous_choir_reach(state: &mut RunState) -> EventOutcome {
    // C# `LuminousChoir.ReachIntoTheFlesh` (`LuminousChoir.cs:60-66`):
    //   1. `CardSelectCmd.FromDeckForRemoval(count: 2)` — pick 2 cards to remove,
    //   2. `CardPileCmd.RemoveFromDeck(cards)`,
    //   3. `AddCurseToDeck<SporeMind>` — add Spore Mind AFTER the removal.
    // Spore Mind is added LAST, so it must NOT be in the removal pool — defer it
    // via `pending_post_removal_cards` (drained when the removal batch resolves).
    // Mirrors `wellspring_bathe` (shared_wave_b.rs).
    use sts2_core::cards::CardId;
    let spore_mind = crate::cards::spec_index::card_pod_from_id(CardId::SporeMind, false);
    if state.interactive_card_removal {
        // Surface a removal `card_select` (the run loop offers RemoveDeckCard per
        // deck card while `pending_deck_removals > 0`); the deferred Spore Mind
        // lands once both picks complete.
        state.pending_deck_removals += 2;
        if let Some(card) = spore_mind {
            state.pending_post_removal_cards.push(card);
        }
    } else {
        // Parity-fuzz / headless auto mode: remove from the deck front + add the
        // curse immediately.
        let _ = state.remove_cards_from_deck(2);
        if let Some(card) = spore_mind {
            state.add_card_to_deck_with_hooks(card, sts2_core::relic_effects::AddSource::EventGain);
        }
    }
    EventOutcome::finished("Reached into the flesh — removed 2 cards and gained Spore Mind.")
}

fn luminous_choir_tribute(state: &mut RunState) -> EventOutcome {
    // C# `OfferTribute`: LoseGold(Gold.IntValue), then
    // `RelicFactory.PullNextRelicFromFront(Owner)` = RollRarity off the player's
    // Rewards stream, front-pop the grab bag (FallbackRelic = Circlet), Obtain.
    let cost = luminous_choir_rolled_cost(state);
    state.gold = state.gold.saturating_sub(cost);
    let rarity = state.roll_reward_relic_rarity();
    let relic = state
        .pop_relic_from_grab_bag(rarity)
        .unwrap_or(LUMINOUS_CHOIR_FALLBACK_RELIC);
    state.grant_event_relic(relic);
    EventOutcome::finished(format!("Paid {cost}g — gained a relic."))
}

const LUMINOUS_CHOIR_OPTIONS: &[EventOptionSpec] = &[
    EventOptionSpec {
        option_id: "reach",
        loc_key: "LUMINOUS_CHOIR.pages.INITIAL.options.REACH_INTO_THE_FLESH",
        default_label: "Reach Into the Flesh",
        default_description: "Remove 2 cards, gain Spore Mind curse.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: luminous_choir_reach,
    },
    EventOptionSpec {
        option_id: "tribute",
        loc_key: "LUMINOUS_CHOIR.pages.INITIAL.options.OFFER_TRIBUTE",
        default_label: "Offer Tribute",
        default_description: "Pay ~125g for a relic.",
        default_vars: &[],
        vars_fn: Some(|s| vec![("Gold", luminous_choir_rolled_cost(s).into())]),
        title_vars_fn: None,
        handler: luminous_choir_tribute,
    },
];

/// `is_allowed`: `luminous_choir_allowed` (`gold >= 149`) — wired.
pub const LUMINOUS_CHOIR: EventSpec = EventSpec {
    id: EventId::LuminousChoir,
    python_name: "LuminousChoir",
    is_allowed: luminous_choir_allowed,
    generate_options: luminous_choir_generate_options,
    options: LUMINOUS_CHOIR_OPTIONS,
};

// ── MorphicGrove ─────────────────────────────────────────────────────

/// `MorphicGrove::group` gold THRESHOLD — `IsAllowed` requires
/// `Gold >= 100`; the Group action itself loses ALL of the player's
/// gold (C# `PlayerCmd.LoseGold(Owner.Gold, …)`), not 100 flat.
/// The Python pin's `act2.py:625` 100-flat drifted from the
/// shipped game. Bead `sts2-rs-e85nx`.
pub const MORPHIC_GROVE_GROUP_COST: i32 = 100;
/// `MorphicGrove::loner` Max HP gain (`act2.py:658`).
pub const MORPHIC_GROVE_LONER_MAX_HP: i32 = 5;
/// `MorphicGrove` Group transforms exactly 2 cards (C# `_transformCount = 2`).
pub const MORPHIC_GROVE_TRANSFORM_COUNT: usize = 2;

/// `MorphicGrove`'s `Id.Entry` (L10N key root) — names its own event stream.
const MORPHIC_GROVE_STREAM: &str = "MORPHIC_GROVE";

/// C# `MorphicGrove.Group` (v0.107.1): `PlayerCmd.LoseGold(Owner.Gold, Stolen)`
/// (ALL gold), then `CardSelectCmd.FromDeckForTransformation(prefs(2))` →
/// `CardCmd.TransformToRandom(item, base.Rng)` per pick. v0.107.1 migrated the
/// transform roll from `base.Owner.RunState.Rng.Niche` (v0.103.2) to the
/// event's own `base.Rng` (`MorphicGrove.cs:50`). Mirrors the WhisperingHollow
/// Hug shape: interactive hosts queue the picks (replacement rolled off the
/// event stream only AFTER each pick — no RNG at queue time); non-interactive
/// hosts (parity-fuzz / random-policy full runs) auto-transform the first 2
/// deck cards off the SAME event stream so the run never strands. The event has
/// no entry-time `base.Rng` roll (no `CalculateVars` override; `MaxHpVar(5)` is
/// a fixed scalar), so transform #N is `base.Rng` read #(N+1) → `skip_reads`
/// equals the loop index.
fn morphic_grove_group(state: &mut RunState) -> EventOutcome {
    let lost = state.gold;
    state.gold = 0;
    if state.interactive_card_transform {
        for skip_reads in 0..MORPHIC_GROVE_TRANSFORM_COUNT {
            state
                .pending_deck_transforms
                .push(sts2_core::PendingDeckTransform {
                    event_stream_roll: Some((MORPHIC_GROVE_STREAM, skip_reads as u32)),
                    ..Default::default()
                });
        }
        EventOutcome::finished(format!("Lost {lost}g (all); pick 2 cards to transform."))
    } else {
        let mut done = 0;
        for skip_reads in 0..MORPHIC_GROVE_TRANSFORM_COUNT {
            let Some(original) = state.master_deck.first().map(|c| c.card_id) else {
                break;
            };
            if let Some(new_id) = crate::cards::spec_index::roll_transform_replacement_event_stream(
                state,
                state.character_id,
                original,
                MORPHIC_GROVE_STREAM,
                skip_reads as u32,
            ) {
                if let Some(card) = crate::cards::spec_index::card_pod_from_id(new_id, false) {
                    state.master_deck.remove(0);
                    state.add_card_to_deck_with_hooks(
                        card,
                        sts2_core::relic_effects::AddSource::EventGain,
                    );
                    done += 1;
                }
            } else {
                break;
            }
        }
        EventOutcome::finished(format!("Lost {lost}g (all); transformed {done} cards."))
    }
}

fn morphic_grove_loner(state: &mut RunState) -> EventOutcome {
    state.gain_max_hp(MORPHIC_GROVE_LONER_MAX_HP);
    EventOutcome::finished(format!("Gained {MORPHIC_GROVE_LONER_MAX_HP} Max HP."))
}

const MORPHIC_GROVE_OPTIONS: &[EventOptionSpec] = &[
    EventOptionSpec {
        option_id: "group",
        loc_key: "MORPHIC_GROVE.pages.INITIAL.options.GROUP",
        default_label: "Group",
        default_description: "Lose all gold, transform 2 cards (unlocked at 100g+).",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: morphic_grove_group,
    },
    EventOptionSpec {
        option_id: "loner",
        loc_key: "MORPHIC_GROVE.pages.INITIAL.options.LONER",
        default_label: "Loner",
        default_description: "Gain 5 Max HP.",
        default_vars: &[(
            "MaxHp",
            EventVarSpec::Int(MORPHIC_GROVE_LONER_MAX_HP as i64),
        )],
        vars_fn: None,
        title_vars_fn: None,
        handler: morphic_grove_loner,
    },
];

/// `is_allowed`: `morphic_grove_allowed` (`gold >= 100`) — wired.
pub const MORPHIC_GROVE: EventSpec = EventSpec {
    id: EventId::MorphicGrove,
    python_name: "MorphicGrove",
    is_allowed: morphic_grove_allowed,
    generate_options: super::default_option_info_from_spec,
    options: MORPHIC_GROVE_OPTIONS,
};

// ── PotionCourier ────────────────────────────────────────────────────

/// `PotionCourier::GrabPotions` Foul-Potion count (C# `DynamicVars["FoulPotions"]
/// = 3m`). The three rewards are deterministic — no RNG — so they queue directly.
pub const POTION_COURIER_FOUL_POTIONS: usize = 3;

fn potion_courier_grab(state: &mut RunState) -> EventOutcome {
    // C# `GrabPotions`: build `FoulPotions` (3) `PotionReward`s and
    // `RewardsCmd.OfferCustom` them. Each reward names FoulPotion explicitly
    // (no pool roll), so queue pre-populated `Reward::Potion`s.
    use sts2_core::{potions::PotionKind, PotionReward, Reward};
    for _ in 0..POTION_COURIER_FOUL_POTIONS {
        state
            .pending_rewards
            .push(Reward::Potion(PotionReward::with_kind(
                PotionKind::FoulPotion,
            )));
    }
    EventOutcome::finished("Grabbed 3 Foul Potions.")
}

fn potion_courier_ransack(state: &mut RunState) -> EventOutcome {
    // C# `Ransack`: ONE `PlayerRng.Rewards.NextItem` over the character++shared
    // potion pool rarity-filtered to Uncommon, offered pre-populated (no rarity
    // float). `NextItem` over an empty sequence returns null without consuming
    // RNG — `roll_event_potion_full_pool_uncommon` returns None likewise — and
    // C# skips the reward in that case (`if potionModel != null`).
    use sts2_core::{PotionReward, Reward};
    if let Some(kind) = crate::potions::all::roll_event_potion_full_pool_uncommon(
        &mut state.rng.rewards,
        state.character_id,
    ) {
        state
            .pending_rewards
            .push(Reward::Potion(PotionReward::with_kind(kind)));
    }
    EventOutcome::finished("Ransacked the courier — gained an uncommon potion.")
}

const POTION_COURIER_OPTIONS: &[EventOptionSpec] = &[
    EventOptionSpec {
        option_id: "grab",
        loc_key: "POTION_COURIER.pages.INITIAL.options.GRAB_POTIONS",
        default_label: "Grab Potions",
        default_description: "Gain 3 Foul Potions.",
        default_vars: &[(
            "FoulPotions",
            EventVarSpec::Int(POTION_COURIER_FOUL_POTIONS as i64),
        )],
        vars_fn: None,
        title_vars_fn: None,
        handler: potion_courier_grab,
    },
    EventOptionSpec {
        option_id: "ransack",
        loc_key: "POTION_COURIER.pages.INITIAL.options.RANSACK",
        default_label: "Ransack",
        default_description: "Gain 1 uncommon potion.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: potion_courier_ransack,
    },
];

/// `is_allowed`: `allowed_act_2_plus` (`current_act_index > 0`) — wired.
/// grab/ransack potion-add wired via `Reward::Potion` (`sts2-rs-m3n0g.4`).
pub const POTION_COURIER: EventSpec = EventSpec {
    id: EventId::PotionCourier,
    python_name: "PotionCourier",
    is_allowed: allowed_act_2_plus,
    generate_options: super::default_option_info_from_spec,
    options: POTION_COURIER_OPTIONS,
};

// ── RanwidTheElder ───────────────────────────────────────────────────

/// `RanwidTheElder::gold` cost (`act2.py:750`).
pub const RANWID_GOLD_COST: i32 = 100;

/// `RanwidTheElder`'s `Id.Entry` (L10N key root) — names its own event stream.
const RANWID_STREAM: &str = "RANWID_THE_ELDER";

/// The potion and the relic Ranwid asks for, in the ORDER C# draws them.
///
/// `GenerateInitialOptions` reads `base.Rng.NextItem(Potions)` first and
/// `base.Rng.NextItem(Relics.Where(IsTradable))` second, so the relic pick sits
/// at cursor 1 — recomputing it means consuming the potion draw first even
/// when only the relic is wanted. Both come back together for that reason.
///
/// Read-only and recomputed: nothing else draws from the stream, so the same
/// pair comes out whether asked while presenting the options or resolving one.
fn ranwid_asks_for(state: &RunState) -> (Option<usize>, Option<RelicId>) {
    let mut rng = state.rng.named(RANWID_STREAM);
    let potion = (!state.potions.is_empty()).then(|| {
        let len = i32::try_from(state.potions.len()).unwrap_or(1);
        usize::try_from(rng.next_int_exclusive(0, len)).unwrap_or(0)
    });
    let tradable: Vec<RelicId> = state
        .relics
        .iter()
        .copied()
        .filter(|&id| crate::relics::tradable::relic_is_tradable_in(state, id))
        .collect();
    let relic = (!tradable.is_empty()).then(|| {
        let len = i32::try_from(tradable.len()).unwrap_or(1);
        tradable[usize::try_from(rng.next_int_exclusive(0, len)).unwrap_or(0)]
    });
    (potion, relic)
}

/// One `RelicFactory.PullNextRelicFromFront(owner)` — roll a rarity off the
/// rewards stream, take the front of that bucket, hand it over.
pub(super) fn ranwid_grant_front_relic(state: &mut RunState) {
    let rarity = state.roll_reward_relic_rarity();
    if let Some(id) = state.pop_relic_from_grab_bag(rarity) {
        state.grant_event_relic(id);
    }
}

fn ranwid_potion(state: &mut RunState) -> EventOutcome {
    // C# `GivePotion`: `PotionCmd.Discard(potion)` then one front pull.
    if let (Some(idx), _) = ranwid_asks_for(state) {
        state.remove_potion_from_belt(idx);
    }
    ranwid_grant_front_relic(state);
    EventOutcome::finished("Gave Ranwid a potion for a relic.")
}

fn ranwid_gold(state: &mut RunState) -> EventOutcome {
    state.gold = state.gold.saturating_sub(RANWID_GOLD_COST);
    ranwid_grant_front_relic(state);
    EventOutcome::finished(format!("Gave Ranwid {RANWID_GOLD_COST}g for a relic."))
}

fn ranwid_relic(state: &mut RunState) -> EventOutcome {
    // C# `GiveRelic`: remove the chosen relic, then TWO front pulls.
    if let (_, Some(id)) = ranwid_asks_for(state) {
        state.lose_relic(id);
    }
    for _ in 0..2 {
        ranwid_grant_front_relic(state);
    }
    EventOutcome::finished("Gave Ranwid a relic for two.")
}

/// `{Potion}` / `{Relic}`: the option TITLES name what Ranwid is asking for
/// ("Give Fire Potion"). Empty when there is nothing to give, so the option
/// falls back to the shorthand rather than printing a bare variable.
fn ranwid_vars(state: &RunState, want_relic: bool) -> Vec<(&'static str, sts2_core::EventVar)> {
    let (potion, relic) = ranwid_asks_for(state);
    if want_relic {
        relic
            .map(|id| {
                (
                    "Relic",
                    sts2_core::EventVar::Text(format!("{}.title", id.to_python_name())),
                )
            })
            .into_iter()
            .collect()
    } else {
        potion
            .map(|idx| {
                (
                    "Potion",
                    sts2_core::EventVar::Text(format!(
                        "{}.title",
                        state.potions[idx].python_name()
                    )),
                )
            })
            .into_iter()
            .collect()
    }
}

const RANWID_OPTIONS: &[EventOptionSpec] = &[
    EventOptionSpec {
        option_id: "potion",
        loc_key: "RANWID_THE_ELDER.pages.INITIAL.options.POTION",
        default_label: "Give a Potion",
        default_description: "Lose a potion, gain a relic.",
        default_vars: &[],
        vars_fn: Some(|s| ranwid_vars(s, false)),
        title_vars_fn: None,
        handler: ranwid_potion,
    },
    EventOptionSpec {
        option_id: "gold",
        loc_key: "RANWID_THE_ELDER.pages.INITIAL.options.GOLD",
        default_label: "Give 100 Gold",
        default_description: "Lose 100g, gain a relic.",
        // C# `RanwidTheElder` binds `Gold` on the option TITLE ("Give {Gold}
        // Gold"), not its description — the only keyed option that does.
        default_vars: &[("Gold", EventVarSpec::Int(RANWID_GOLD_COST as i64))],
        vars_fn: None,
        title_vars_fn: None,
        handler: ranwid_gold,
    },
    EventOptionSpec {
        option_id: "relic",
        loc_key: "RANWID_THE_ELDER.pages.INITIAL.options.RELIC",
        default_label: "Give a Relic",
        default_description: "Lose a relic, gain 2 relics.",
        default_vars: &[],
        vars_fn: Some(|s| ranwid_vars(s, true)),
        title_vars_fn: None,
        handler: ranwid_relic,
    },
];

/// `is_allowed`: `ranwid_the_elder_allowed`
/// (`current_act_index > 0 && gold >= 100 && held_potions && relics`) — wired.
pub const RANWID_THE_ELDER: EventSpec = EventSpec {
    id: EventId::RanwidTheElder,
    python_name: "RanwidTheElder",
    is_allowed: ranwid_the_elder_allowed,
    generate_options: super::default_option_info_from_spec,
    options: RANWID_OPTIONS,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn rs_with_gold(gold: i32) -> RunState {
        let mut rs = RunState::new();
        rs.gold = gold;
        rs
    }

    // ── DollRoom multi-page (m3n0g.3) ──────────────────────────────

    /// Page 0 presents only the three approaches (`DollRoom.cs:84`), not the
    /// three page-1 doll picks that share the static option list.
    #[test]
    fn doll_room_initial_page_offers_three_approaches() {
        let rs = RunState::new();
        let opts = doll_room_generate_options(&rs, &DOLL_ROOM);
        let ids: Vec<_> = opts.iter().map(|o| o.option_id).collect();
        assert_eq!(ids, ["random", "take_time", "examine"]);
    }

    #[test]
    fn doll_room_random_obtains_one_doll_and_finishes() {
        let mut rs = RunState::new();
        rs.direct_event_relic_grant = true;
        let before = rs.relics.len();
        let out = doll_room_random(&mut rs);
        assert!(
            out.finished,
            "random doll finishes immediately (single page)"
        );
        assert_eq!(rs.relics.len(), before + 1, "exactly one doll obtained");
        let granted = *rs.relics.last().unwrap();
        assert!(
            DOLL_ROOM_DOLLS.iter().any(|(r, _)| *r == granted),
            "granted relic {granted:?} must be one of the three dolls",
        );
        // The roll is a pure function of the run seed.
        let mut rs2 = RunState::new();
        rs2.direct_event_relic_grant = true;
        let _ = doll_room_random(&mut rs2);
        assert_eq!(rs2.relics.last(), Some(&granted), "roll is deterministic");
    }

    #[test]
    fn doll_room_take_time_opens_two_doll_page_without_finishing() {
        let mut rs = RunState::new();
        rs.current_hp = 50;
        let out = doll_room_take_time(&mut rs);
        assert!(!out.finished, "take-time lingers for the doll-pick page");
        assert_eq!(rs.current_hp, 50 - DOLL_ROOM_TAKE_TIME_DAMAGE);
        assert_eq!(rs.event_state.page, 1, "advanced to the pick page");
        let opts = doll_room_generate_options(&rs, &DOLL_ROOM);
        assert_eq!(opts.len(), 2, "Take Some Time offers 2 dolls");
        assert!(opts.iter().all(|o| o.enabled));
    }

    /// `StableShuffle` sorts the dolls by relic id entry (ordinal) before the
    /// Fisher-Yates pass (`DollRoom.cs:28`, same in both builds), so every
    /// seed shuffles [BingBong, Daughter, MrStruggles], not the declaration
    /// order (GH #806).
    #[test]
    fn doll_room_take_time_shuffles_in_entry_order_both_versions() {
        for (seed, version) in (0..32).flat_map(|seed| {
            [
                sts2_core::GameVersion::V0_107_1,
                sts2_core::GameVersion::V0_111_0,
            ]
            .map(|v| (seed, v))
        }) {
            let mut rs = RunState::new();
            rs.game_version = version;
            rs.rng = sts2_core::run_rng_set::RunRngSet::new_for(rs.game_version, seed);
            rs.current_hp = 50;
            let mut expected: Vec<i16> = vec![2, 0, 1];
            doll_room_rng(&rs).shuffle(&mut expected);
            expected.truncate(2);
            let _ = doll_room_take_time(&mut rs);
            assert_eq!(rs.event_state.scratch, expected, "seed {seed} {version:?}");
        }
    }

    #[test]
    fn doll_room_examine_opens_three_doll_page_without_finishing() {
        let mut rs = RunState::new();
        rs.current_hp = 60;
        let out = doll_room_examine(&mut rs);
        assert!(!out.finished);
        assert_eq!(rs.current_hp, 60 - DOLL_ROOM_EXAMINE_DAMAGE);
        assert_eq!(rs.event_state.page, 1);
        let opts = doll_room_generate_options(&rs, &DOLL_ROOM);
        assert_eq!(opts.len(), 3, "Examine offers all 3 dolls");
    }

    #[test]
    fn doll_room_page1_pick_grants_the_chosen_doll_and_finishes() {
        let mut rs = RunState::new();
        rs.direct_event_relic_grant = true;
        rs.event_state.page = 1;
        let before = rs.relics.len();
        let out = doll_room_pick_mr_struggles(&mut rs);
        assert!(out.finished, "picking a doll finishes the event");
        assert_eq!(rs.relics.len(), before + 1);
        assert!(rs.relics.contains(&RelicId::MrStruggles));
    }

    // ── CrystalSphere (GH #817) ────────────────────────────────────

    /// `CalculateVars`: `UncoverFutureCost = 50 + Rng.NextInt(1, 50)`, the
    /// FIRST draw on the event's own `CRYSTAL_SPHERE` stream.
    #[test]
    fn crystal_sphere_cost_is_fifty_plus_first_event_rng_draw() {
        for seed in 0..20u32 {
            let mut rs = RunState::new();
            rs.rng = sts2_core::RunRngSet::new(seed as i32 * 7919 + 13);
            let mut rng = rs.rng.named("CRYSTAL_SPHERE");
            let want = 50 + rng.next_int_exclusive(1, 50);
            let cost = crystal_sphere_pay_cost(&rs);
            assert_eq!(cost, want);
            assert!((51..=99).contains(&cost));
        }
    }

    /// Uncover Future spends the rolled cost (and only gains gold back from
    /// revealed gold items, 10 or 30 each).
    #[test]
    fn crystal_sphere_pay_deducts_rolled_cost() {
        let mut rs = rs_with_gold(200);
        let cost = crystal_sphere_pay_cost(&rs);
        let outcome = (CRYSTAL_SPHERE.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        let gained = rs.gold - (200 - cost);
        assert!(
            (0..=5 * 10 + 2 * 30).contains(&gained) && gained % 10 == 0,
            "gold after paying {cost}: {}",
            rs.gold
        );
        assert!(
            !rs.master_deck
                .iter()
                .any(|c| c.card_id == sts2_core::cards::CardId::Debt),
            "no Debt on Uncover Future"
        );
    }

    /// Payment Plan adds Debt first and charges no gold.
    #[test]
    fn crystal_sphere_debt_adds_debt_curse_without_spending_gold() {
        use sts2_core::cards::CardId;
        let mut rs = rs_with_gold(50);
        let deck_before = rs.master_deck.len();
        let outcome = (CRYSTAL_SPHERE.options[1].handler)(&mut rs);
        assert!(outcome.finished);
        assert!(rs.gold >= 50, "Payment Plan charges no gold");
        assert_eq!(
            rs.master_deck[deck_before].card_id,
            CardId::Debt,
            "Debt first"
        );
    }

    /// C# offers both options with no gold gate; the cost var carries the roll.
    #[test]
    fn crystal_sphere_options_are_ungated_and_carry_the_cost() {
        let rs = rs_with_gold(0);
        let opts = (CRYSTAL_SPHERE.generate_options)(&rs, &CRYSTAL_SPHERE);
        assert_eq!(opts.len(), 2);
        assert!(opts.iter().all(|o| o.enabled));
        let cost = crystal_sphere_pay_cost(&rs);
        assert!(opts[0].vars.contains(&(
            "UncoverFutureCost",
            sts2_core::EventVar::Int(i64::from(cost))
        )));
    }

    // ── DollRoom ───────────────────────────────────────────────────

    #[test]
    fn doll_room_random_queues_relic_reward() {
        // Without `direct_event_relic_grant`, grant_event_relic defers to
        // the relic-reward queue (the run-loop drains it).
        let mut rs = RunState::new();
        let outcome = (DOLL_ROOM.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.relic_rewards.len(), 1);
    }

    // (take_time / examine no longer finish-and-queue — they open the
    // multi-page doll-pick sub-page; see the multi-page tests above.)

    // ── EndlessConveyor ────────────────────────────────────────────

    /// Put the belt at `[num_grabs, last = current = dish]` with the event
    /// stream `draws` reads in, as if that many draws had happened.
    fn conveyor_at(rs: &mut RunState, num_grabs: i16, dish: ConveyorDish, draws: u32) {
        let mut rng = rs.rng.named(ENDLESS_CONVEYOR_STREAM);
        for _ in 0..draws {
            let _ = rng.next_int(0, 1);
        }
        rs.event_state.scratch = vec![num_grabs, dish as i16, dish as i16];
        rs.event_state.rng = Some(rng);
    }

    fn conveyor_option_ids(rs: &RunState) -> Vec<&'static str> {
        endless_conveyor_generate_options(rs, &ENDLESS_CONVEYOR)
            .iter()
            .map(|o| o.option_id)
            .collect()
    }

    fn fill_potion_belt(rs: &mut RunState) {
        while rs.potions.len() < rs.potion_belt_cap() {
            rs.potions.push(sts2_core::potions::PotionKind::FirePotion);
        }
    }

    /// `CalculateVars` rolls the first dish at event start (one event-stream
    /// draw) and binding again does not re-roll; page INITIAL is
    /// [grab, observe].
    #[test]
    fn endless_conveyor_event_start_rolls_one_dish() {
        let mut rs = rs_with_gold(200);
        super::super::dispatch::begin_event(&mut rs, EventId::EndlessConveyor);
        assert_eq!(rs.event_state.scratch.len(), 3);
        assert_eq!(rs.event_state.scratch[0], 1, "NumOfGrabs");
        assert_eq!(rs.event_state.rng.as_ref().unwrap().counter(), 1);
        endless_conveyor_before_event_started(&mut rs);
        assert_eq!(rs.event_state.rng.as_ref().unwrap().counter(), 1);
        assert_eq!(conveyor_option_ids(&rs), ["grab", "observe"]);
    }

    /// The pool (EndlessConveyor.cs:219-238): four base dishes, Suspicious
    /// Condiment only with an open potion slot, Clam Roll only below max HP,
    /// Golden Fysh only after the first roll, and never the last dish.
    #[test]
    fn endless_conveyor_dish_pool_composition() {
        use ConveyorDish as D;
        let mut rs = rs_with_gold(200);
        rs.current_hp = rs.max_hp;
        fill_potion_belt(&mut rs);
        let first = ConveyorBelt {
            num_grabs: 1,
            last: CONVEYOR_NO_DISH,
            current: CONVEYOR_NO_DISH,
        };
        assert_eq!(
            conveyor_dish_pool(&rs, &first),
            [D::Caviar, D::SpicySnappy, D::JellyLiver, D::FriedEel]
        );
        rs.potions.clear();
        rs.current_hp = rs.max_hp - 1;
        let later = ConveyorBelt {
            num_grabs: 2,
            last: D::SpicySnappy as i16,
            current: D::SpicySnappy as i16,
        };
        assert_eq!(
            conveyor_dish_pool(&rs, &later),
            [
                D::Caviar,
                D::JellyLiver,
                D::FriedEel,
                D::SuspiciousCondiment,
                D::ClamRoll,
                D::GoldenFysh
            ]
        );
    }

    /// The pick is float math (EndlessConveyor.cs:247-261): seed 88495220's
    /// draw 0.3999999985 is 0.4f, and 0.4f * 15f rounds to 6.0f, which is
    /// not `< 6` (Caviar's running sum), so C# serves Spicy Snappy. A double
    /// roll (5.99999998) would have served Caviar.
    #[test]
    fn endless_conveyor_roll_dish_is_float_math() {
        let mut rs = rs_with_gold(200);
        rs.current_hp = rs.max_hp;
        fill_potion_belt(&mut rs);
        let mut belt = ConveyorBelt {
            num_grabs: 0,
            last: CONVEYOR_NO_DISH,
            current: CONVEYOR_NO_DISH,
        };
        let mut rng = sts2_core::dotnet_rng::Rng::from_seed(88_495_220);
        conveyor_roll_dish(&rs, &mut belt, &mut rng);
        assert_eq!(belt.current, ConveyorDish::SpicySnappy as i16);
        assert_eq!(belt.last, ConveyorDish::SpicySnappy as i16);
    }

    /// Every 5th roll is Seapunk Salad and draws nothing
    /// (EndlessConveyor.cs:214-220).
    #[test]
    fn endless_conveyor_fifth_roll_is_seapunk_without_a_draw() {
        let rs = rs_with_gold(200);
        let mut belt = ConveyorBelt {
            num_grabs: 4,
            last: ConveyorDish::Caviar as i16,
            current: ConveyorDish::Caviar as i16,
        };
        let mut rng = rs.rng.named(ENDLESS_CONVEYOR_STREAM);
        conveyor_roll_dish(&rs, &mut belt, &mut rng);
        assert_eq!(belt.current, ConveyorDish::SeapunkSalad as i16);
        assert_eq!(rng.counter(), 0);
    }

    /// Grab pays 40, eats the dish, rolls the next one and opens page
    /// GRAB_SOMETHING_OFF_THE_BELT = [grab, leave] (EndlessConveyor.cs:121-135).
    #[test]
    fn endless_conveyor_grab_caviar_pays_and_continues() {
        let mut rs = rs_with_gold(200);
        let max_before = rs.max_hp;
        conveyor_at(&mut rs, 1, ConveyorDish::Caviar, 1);
        let outcome = (ENDLESS_CONVEYOR.options[0].handler)(&mut rs);
        assert!(!outcome.finished);
        assert_eq!(rs.gold, 200 - ENDLESS_CONVEYOR_GRAB_COST);
        assert_eq!(rs.max_hp, max_before + ENDLESS_CONVEYOR_CAVIAR_MAX_HP);
        assert_eq!(rs.event_state.page, 1);
        assert_eq!(rs.event_state.scratch[0], 2, "NumOfGrabs");
        assert_ne!(rs.event_state.scratch[2], ConveyorDish::Caviar as i16);
        assert_eq!(rs.event_state.rng.as_ref().unwrap().counter(), 2);
        assert_eq!(conveyor_option_ids(&rs), ["grab", "leave"]);
        let leave = (ENDLESS_CONVEYOR.options[2].handler)(&mut rs);
        assert!(leave.finished);
    }

    /// Golden Fysh is free and pays 75 (EndlessConveyor.cs:123-126,193-196).
    #[test]
    fn endless_conveyor_golden_fysh_is_free() {
        let mut rs = rs_with_gold(200);
        conveyor_at(&mut rs, 2, ConveyorDish::GoldenFysh, 1);
        let _ = (ENDLESS_CONVEYOR.options[0].handler)(&mut rs);
        assert_eq!(rs.gold, 200 + ENDLESS_CONVEYOR_GOLDEN_FYSH_GOLD);
    }

    /// Below 40 gold the grab row is LOCKED, even for Golden Fysh
    /// (EndlessConveyor.cs:137-144).
    #[test]
    fn endless_conveyor_grab_locked_below_cost() {
        let mut rs = rs_with_gold(39);
        conveyor_at(&mut rs, 2, ConveyorDish::GoldenFysh, 1);
        rs.event_state.page = 1;
        let opts = endless_conveyor_generate_options(&rs, &ENDLESS_CONVEYOR);
        assert_eq!(opts[0].option_id, "grab");
        assert!(!opts[0].enabled);
        assert_eq!(opts[0].loc_key, "ENDLESS_CONVEYOR.pages.ALL.options.LOCKED");
        assert!(opts[1].enabled);
    }

    /// The locked grab cannot be taken, so its preview shows nothing: resolving
    /// it on the preview clone ate the hidden dish and charged the gold, and
    /// the preview moved with the event seed.
    #[test]
    fn endless_conveyor_locked_grab_previews_nothing() {
        let mut rs = rs_with_gold(200);
        super::super::dispatch::begin_event(&mut rs, EventId::EndlessConveyor);
        conveyor_at(&mut rs, 2, ConveyorDish::Caviar, 1);
        rs.event_state.page = 1;
        rs.gold = 39;
        let d = super::super::dispatch::preview_option_effect(&rs, 0).expect("preview");
        assert_eq!((d.gold_delta, d.hp_delta, d.max_hp_delta), (0, 0, 0));
    }

    /// Seapunk Salad adds Feeding Frenzy (EndlessConveyor.cs:180-184).
    #[test]
    fn endless_conveyor_seapunk_salad_adds_feeding_frenzy() {
        crate::cards::dispatch::ensure_installed();
        let mut rs = rs_with_gold(200);
        conveyor_at(&mut rs, 5, ConveyorDish::SeapunkSalad, 4);
        let _ = (ENDLESS_CONVEYOR.options[0].handler)(&mut rs);
        assert!(rs
            .master_deck
            .iter()
            .any(|c| c.card_id == sts2_core::cards::CardId::FeedingFrenzyCard));
        assert_eq!(rs.gold, 200 - ENDLESS_CONVEYOR_GRAB_COST);
    }

    /// Suspicious Condiment claims its potion (one Rewards read) before the
    /// next `RollDish` reads `HasOpenPotionSlots` (EndlessConveyor.cs:150-161).
    #[test]
    fn endless_conveyor_condiment_claims_the_potion_first() {
        let mut rs = rs_with_gold(200);
        rs.potions.clear();
        let cap = rs.potion_belt_cap();
        for _ in 1..cap {
            rs.potions.push(sts2_core::potions::PotionKind::FirePotion);
        }
        let rewards_before = rs.rng.rewards.counter();
        conveyor_at(&mut rs, 1, ConveyorDish::SuspiciousCondiment, 1);
        let _ = (ENDLESS_CONVEYOR.options[0].handler)(&mut rs);
        assert_eq!(rs.potions.len(), cap, "belt now full");
        assert_eq!(rs.rng.rewards.counter(), rewards_before + 1);
    }

    /// Jelly Liver queues the transform pick at the current event-stream
    /// position and reserves its one draw ahead of the next `RollDish`
    /// (EndlessConveyor.cs:163-170).
    #[test]
    fn endless_conveyor_jelly_liver_reserves_the_transform_draw() {
        crate::cards::dispatch::ensure_installed();
        let mut rs = rs_with_gold(200);
        rs.interactive_card_transform = true;
        rs.master_deck.push(
            crate::cards::spec_index::card_pod_from_id(
                sts2_core::cards::CardId::StrikeIronclad,
                false,
            )
            .unwrap(),
        );
        conveyor_at(&mut rs, 1, ConveyorDish::JellyLiver, 1);
        let _ = (ENDLESS_CONVEYOR.options[0].handler)(&mut rs);
        assert_eq!(rs.pending_deck_transforms.len(), 1);
        assert_eq!(
            rs.pending_deck_transforms[0].event_stream_roll,
            Some((ENDLESS_CONVEYOR_STREAM, 1))
        );
        assert_eq!(rs.event_state.rng.as_ref().unwrap().counter(), 3);
    }

    #[test]
    fn endless_conveyor_observe_upgrades_one_card_without_spending_gold() {
        crate::cards::dispatch::ensure_installed();
        use sts2_core::cards::CardId;
        let mut rs = rs_with_gold(100);
        rs.rng.seed = 7777;
        // Three upgradable cards; exactly one should end up upgraded.
        for _ in 0..3 {
            rs.master_deck.push(
                crate::cards::spec_index::card_pod_from_id(CardId::StrikeIronclad, false).unwrap(),
            );
        }
        let upgraded_before = rs.master_deck.iter().filter(|c| c.upgraded).count();
        assert_eq!(upgraded_before, 0);
        let outcome = (ENDLESS_CONVEYOR.options[1].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.gold, 100, "observe is free");
        let upgraded_after = rs.master_deck.iter().filter(|c| c.upgraded).count();
        assert_eq!(upgraded_after, 1, "exactly one card upgraded");
        assert_eq!(rs.master_deck.len(), 3, "deck size unchanged");
    }

    #[test]
    fn endless_conveyor_observe_noop_when_no_upgradable_card() {
        crate::cards::dispatch::ensure_installed();
        let mut rs = rs_with_gold(100);
        // Empty deck → C# `enumerable2.Any()` is false, the upgrade is skipped.
        let outcome = (ENDLESS_CONVEYOR.options[1].handler)(&mut rs);
        assert!(outcome.finished);
        assert!(rs.master_deck.is_empty());
    }

    #[test]
    fn endless_conveyor_observe_pick_is_deterministic_under_same_seed() {
        crate::cards::dispatch::ensure_installed();
        use sts2_core::cards::CardId;
        let build = || {
            let mut rs = rs_with_gold(100);
            rs.rng.seed = 31415;
            for _ in 0..6 {
                rs.master_deck.push(
                    crate::cards::spec_index::card_pod_from_id(CardId::StrikeIronclad, false)
                        .unwrap(),
                );
            }
            rs
        };
        let mut a = build();
        let mut b = build();
        (ENDLESS_CONVEYOR.options[1].handler)(&mut a);
        (ENDLESS_CONVEYOR.options[1].handler)(&mut b);
        let upgraded_idx = |rs: &RunState| {
            rs.master_deck
                .iter()
                .position(|c| c.upgraded)
                .expect("one card upgraded")
        };
        assert_eq!(
            upgraded_idx(&a),
            upgraded_idx(&b),
            "same seed ⇒ same upgrade target (RollDish NextFloat + NextItem replay)"
        );
    }

    // ── FakeMerchant ───────────────────────────────────────────────

    #[test]
    fn fake_merchant_buy_deducts_50g_and_queues_relic() {
        let mut rs = rs_with_gold(200);
        let outcome = (FAKE_MERCHANT.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.gold, 200 - FAKE_MERCHANT_BUY_COST);
        assert_eq!(rs.relic_rewards.len(), 1);
    }

    /// C# rolls each stocked slot's price off the shared `Shops` stream at
    /// event start (`MerchantRelicEntry.cs:45`): 6 draws, and the buy
    /// charges the rolled price.
    #[test]
    fn fake_merchant_event_start_rolls_six_shop_prices() {
        let mut rs = rs_with_gold(200);
        let mut expected = rs.rng.shops.clone();
        for _ in 0..6 {
            let _ = expected.next_float_range(0.85, 1.15);
        }
        fake_merchant_before_event_started(&mut rs);
        assert_eq!(rs.rng.shops.counter(), expected.counter());
        assert_eq!(rs.event_state.scratch.len(), 6);
        assert!(rs
            .event_state
            .scratch
            .iter()
            .all(|&p| (43..=57).contains(&p)));
        // Idempotent: binding the same visit again draws nothing.
        fake_merchant_before_event_started(&mut rs);
        assert_eq!(rs.rng.shops.counter(), expected.counter());
        let price = i32::from(rs.event_state.scratch[0]);
        let _ = (FAKE_MERCHANT.options[0].handler)(&mut rs);
        assert_eq!(rs.gold, 200 - price);
    }

    /// `MerchantRelicEntry.cs:45` prices in float: the first `Shops` draw of
    /// seed 49287 is 0.59999989 → jitter 1.02999997 (f64) whose f32 is
    /// 1.03f, and 50 * 1.03f = 51.5f banker-rounds to 52. The f64 product
    /// 51.4999984 gave 51.
    #[test]
    fn fake_merchant_price_is_float_math() {
        let mut rs = rs_with_gold(200);
        rs.rng.shops = sts2_core::dotnet_rng::Rng::from_seed(49287);
        fake_merchant_before_event_started(&mut rs);
        assert_eq!(rs.event_state.scratch[0], 52);
    }

    #[test]
    fn fake_merchant_throw_needs_a_held_foul_potion() {
        // GH #885: without a Foul Potion there is nothing to throw, so the
        // fight is not on offer; the slot stays put (buy, throw_foul, leave).
        let rs = rs_with_gold(200);
        let opts = fake_merchant_generate_options(&rs, &FAKE_MERCHANT);
        let ids: Vec<_> = opts.iter().map(|o| o.option_id).collect();
        assert_eq!(ids, ["buy", "throw_foul", "leave"]);
        assert!(opts[0].enabled);
        assert!(!opts[1].enabled, "no Foul Potion held");
        assert!(opts[2].enabled);

        let mut rs = rs_with_gold(200);
        rs.potions.push(sts2_core::potions::PotionKind::FoulPotion);
        let opts = fake_merchant_generate_options(&rs, &FAKE_MERCHANT);
        assert!(opts[1].enabled, "Foul Potion held");
    }

    #[test]
    fn fake_merchant_buy_needs_50_gold() {
        let opts = fake_merchant_generate_options(&rs_with_gold(49), &FAKE_MERCHANT);
        assert!(!opts[0].enabled, "49g cannot buy");
        let opts = fake_merchant_generate_options(&rs_with_gold(50), &FAKE_MERCHANT);
        assert!(opts[0].enabled, "50g can buy");
    }

    #[test]
    fn fake_merchant_leave_is_finished_noop() {
        let mut rs = rs_with_gold(200);
        let outcome = (FAKE_MERCHANT.options[2].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.gold, 200);
    }

    // ── FieldOfManSizedHoles ───────────────────────────────────────

    #[test]
    fn field_of_man_sized_holes_resist_removes_two_and_adds_normality() {
        use sts2_core::cards::CardId;
        let mut rs = RunState::new();
        for _ in 0..3 {
            rs.master_deck.push(
                crate::cards::spec_index::card_pod_from_id(CardId::StrikeIronclad, false).unwrap(),
            );
        }
        let deck_before = rs.master_deck.len();
        let outcome = (FIELD_OF_MAN_SIZED_HOLES.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        // Net: −2 removed, +1 Normality = −1.
        assert_eq!(rs.master_deck.len(), deck_before - 1);
        assert!(rs
            .master_deck
            .iter()
            .any(|c| c.card_id == CardId::Normality));
    }

    #[test]
    fn field_of_man_sized_holes_enter_enchants_a_card_with_perfect_fit() {
        use sts2_core::cards::{CardId, EnchantmentId};
        let mut rs = RunState::new();
        rs.master_deck.push(
            crate::cards::spec_index::card_pod_from_id(CardId::StrikeIronclad, false).unwrap(),
        );
        let outcome = (FIELD_OF_MAN_SIZED_HOLES.options[1].handler)(&mut rs);
        assert!(outcome.finished);
        assert!(rs.master_deck.iter().any(|c| c
            .enchantments
            .iter()
            .any(|(e, _)| e == EnchantmentId::PerfectFit)));
    }

    // ── JungleMazeAdventure ────────────────────────────────────────

    #[test]
    fn jungle_maze_solo_costs_hp_and_grants_rolled_gold() {
        let mut rs = RunState::new();
        let (solo_gold, _join) = jungle_maze_gold(&rs);
        // The per-event RNG swing keeps the reward in [135, 165].
        assert!(
            (135..=165).contains(&solo_gold),
            "solo gold {solo_gold} within the ±15 swing"
        );
        let (hp_before, gold_before) = (rs.current_hp, rs.gold);
        let outcome = (JUNGLE_MAZE_ADVENTURE.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.current_hp, hp_before - JUNGLE_MAZE_SOLO_DAMAGE);
        assert_eq!(rs.gold, gold_before + solo_gold);
    }

    #[test]
    fn jungle_maze_join_grants_rolled_gold_safe() {
        let mut rs = RunState::new();
        let (_solo, join_gold) = jungle_maze_gold(&rs);
        assert!(
            (35..=65).contains(&join_gold),
            "join gold {join_gold} within the ±15 swing"
        );
        let (hp_before, gold_before) = (rs.current_hp, rs.gold);
        let outcome = (JUNGLE_MAZE_ADVENTURE.options[1].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.current_hp, hp_before);
        assert_eq!(rs.gold, gold_before + join_gold);
    }

    #[test]
    fn csharp_decimal_from_f32_keeps_seven_significant_digits() {
        let same = |x: f32, (wn, wd): (i128, i128)| {
            let (n, d) = csharp_decimal_from_f32(x);
            assert_eq!(n * wd, wn * d, "{x} -> {n}/{d}, want {wn}/{wd}");
        };
        same(0.0, (0, 1));
        same(10.0, (10, 1));
        same(-12.4, (-124, 10));
        // 1/3 as f32 is 0.3333333432..., which C# converts to 0.3333333.
        same(1.0 / 3.0, (3_333_333, 10_000_000));
        // 14.999999 as f32 rounds to 7 digits: 15.00000, not 14.99999.
        same(14.999_999, (15, 1));
    }

    /// C# `PlayerCmd.GainGold` multiplies the DECIMAL `SoloGold` by Bowler
    /// Hat's 1.25 and truncates once. Truncating the roll first loses a gold
    /// whenever the fraction carries the product over an integer.
    #[test]
    fn jungle_maze_bowler_hat_truncates_the_decimal_once() {
        let mut saw_double_truncation_gap = false;
        for seed in 0..64 {
            for opt in 0..2 {
                let mut rs = RunState::new();
                rs.rng = sts2_core::run_rng_set::RunRngSet::new(seed);
                rs.relics.push(RelicId::BowlerHat);
                let (solo, join) = jungle_maze_gold_exact(&rs);
                let (n, d) = if opt == 0 { solo } else { join };
                let want = i32::try_from(n * 5 / (d * 4)).unwrap();
                let truncated_first = i32::try_from(n / d).unwrap() * 5 / 4;
                saw_double_truncation_gap |= want != truncated_first;
                let before = rs.gold;
                let _ = (JUNGLE_MAZE_ADVENTURE.options[opt].handler)(&mut rs);
                assert_eq!(rs.gold - before, want, "seed {seed} option {opt}");
            }
        }
        assert!(
            saw_double_truncation_gap,
            "the sweep must cover a carry case"
        );
    }

    #[test]
    fn jungle_maze_gold_is_deterministic_per_seed() {
        // Same run seed → identical per-event roll (fresh stream keyed by
        // base seed + hash("JUNGLE_MAZE_ADVENTURE")); different seeds vary.
        let mut a = RunState::new();
        a.rng = sts2_core::run_rng_set::RunRngSet::new(7);
        let mut b = RunState::new();
        b.rng = sts2_core::run_rng_set::RunRngSet::new(7);
        let mut c = RunState::new();
        c.rng = sts2_core::run_rng_set::RunRngSet::new(8);
        assert_eq!(jungle_maze_gold(&a), jungle_maze_gold(&b));
        assert_ne!(jungle_maze_gold(&a), jungle_maze_gold(&c));
    }

    // ── LuminousChoir ──────────────────────────────────────────────

    #[test]
    fn luminous_choir_reach_interactive_queues_two_removals_and_defers_spore_mind() {
        use sts2_core::cards::CardId;
        let mut rs = RunState::new();
        rs.interactive_card_removal = true;
        let deck_before = rs.master_deck.len();
        let outcome = (LUMINOUS_CHOIR.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(
            rs.pending_deck_removals, 2,
            "surfaces a 2-card removal select"
        );
        assert_eq!(rs.master_deck.len(), deck_before, "no eager deck mutation");
        assert_eq!(
            rs.pending_post_removal_cards.len(),
            1,
            "Spore Mind deferred"
        );
        assert_eq!(
            rs.pending_post_removal_cards[0].card_id,
            CardId::SporeMind,
            "the deferred curse is Spore Mind"
        );
    }

    #[test]
    fn luminous_choir_reach_auto_removes_two_and_adds_spore_mind() {
        use sts2_core::cards::CardId;
        let mut rs = RunState::new(); // interactive_card_removal defaults off
                                      // Seed a deck so the 2-card front removal has cards to take.
        for _ in 0..4 {
            if let Some(c) =
                crate::cards::spec_index::card_pod_from_id(CardId::StrikeNecrobinder, false)
            {
                rs.master_deck.push(c);
            }
        }
        let deck_before = rs.master_deck.len();
        let outcome = (LUMINOUS_CHOIR.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(
            rs.pending_deck_removals, 0,
            "auto mode does not surface a select"
        );
        // -2 removed, +1 Spore Mind = net -1.
        assert_eq!(rs.master_deck.len(), deck_before - 1);
        assert!(
            rs.master_deck
                .iter()
                .any(|c| c.card_id == CardId::SporeMind),
            "Spore Mind added in auto mode"
        );
    }

    #[test]
    fn luminous_choir_tribute_deducts_cost_and_queues_relic() {
        let mut rs = rs_with_gold(200);
        // Cost is rolled from the event's private stream (149 - NextInt(0,50));
        // re-derive the same value to assert the deduction.
        let expected_cost = luminous_choir_rolled_cost(&rs);
        assert!((100..=149).contains(&expected_cost));
        let outcome = (LUMINOUS_CHOIR.options[1].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.gold, 200 - expected_cost);
        assert_eq!(rs.relic_rewards.len(), 1);
    }

    #[test]
    fn luminous_choir_cost_in_100_149_across_seeds() {
        // C# `149 - NextInt(0, 50)` rolls `[0, 49]` → cost `[100, 149]`. The
        // inclusive-vs-exclusive bug (`next_int(0, 50)` = `[0, 50]`) let the
        // cost reach 99, charging 1 less than C#. Sweep seeds to prove the
        // floor is 100 (a single bad seed would catch a regression). Bead
        // `sts2-rs-RP14`.
        for seed in 0..400u32 {
            let mut rs = rs_with_gold(500);
            rs.rng = sts2_core::run_rng_set::RunRngSet::new(seed as i32);
            let cost = luminous_choir_rolled_cost(&rs);
            assert!(
                (100..=149).contains(&cost),
                "seed {seed}: tribute cost {cost} outside [100, 149]"
            );
        }
    }

    #[test]
    fn luminous_choir_options_disable_tribute_when_broke() {
        let rs = rs_with_gold(0);
        let opts = (LUMINOUS_CHOIR.generate_options)(&rs, &LUMINOUS_CHOIR);
        assert_eq!(opts.len(), 2);
        assert!(opts[0].enabled);
        assert!(!opts[1].enabled);
    }

    // ── MorphicGrove ───────────────────────────────────────────────

    #[test]
    fn morphic_grove_group_zeroes_all_gold() {
        // C# `PlayerCmd.LoseGold(Owner.Gold, ...)` subtracts the
        // player's CURRENT gold, netting to zero regardless of the
        // 100g threshold. Bead `sts2-rs-e85nx`.
        let mut rs = rs_with_gold(200);
        let outcome = (MORPHIC_GROVE.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.gold, 0);
    }

    #[test]
    fn morphic_grove_loner_gains_5_max_hp() {
        let mut rs = RunState::new();
        let max_before = rs.max_hp;
        let outcome = (MORPHIC_GROVE.options[1].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.max_hp, max_before + MORPHIC_GROVE_LONER_MAX_HP);
    }

    // ── PotionCourier ──────────────────────────────────────────────

    #[test]
    fn potion_courier_grab_queues_three_foul_potions() {
        use sts2_core::{potions::PotionKind, PotionReward, Reward};
        let mut rs = RunState::new();
        let outcome = (POTION_COURIER.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.pending_rewards.len(), POTION_COURIER_FOUL_POTIONS);
        assert!(rs.pending_rewards.iter().all(|r| matches!(
            r,
            Reward::Potion(p) if *p == PotionReward::with_kind(PotionKind::FoulPotion)
        )));
    }

    #[test]
    fn potion_courier_ransack_queues_one_uncommon_potion() {
        use sts2_core::{potions::PotionRarity, Reward};
        let mut rs = RunState::new();
        let outcome = (POTION_COURIER.options[1].handler)(&mut rs);
        assert!(outcome.finished);
        // Ironclad's pool always has Uncommons, so exactly one reward queues.
        assert_eq!(rs.pending_rewards.len(), 1);
        let Reward::Potion(p) = &rs.pending_rewards[0] else {
            panic!("expected a Potion reward");
        };
        let kind = p.potion_kind.expect("ransack pre-rolls the potion kind");
        assert_eq!(
            crate::potions::all::spec_for(kind).rarity,
            PotionRarity::Uncommon,
        );
    }

    // ── RanwidTheElder ─────────────────────────────────────────────

    /// A stocked Ranwid: two potions, three tradable relics, a full bag.
    fn ranwid_state() -> RunState {
        use sts2_core::potions::PotionKind;
        let mut rs = rs_with_gold(200);
        rs.act_index = 2;
        rs.direct_event_relic_grant = true;
        rs.potions = vec![PotionKind::BloodPotion, PotionKind::FirePotion];
        rs.potion_slot_indices = vec![0, 1];
        rs.relics = vec![RelicId::Anchor, RelicId::BloodVial, RelicId::HappyFlower];
        rs.relic_grab_bag_by_rarity.insert(
            sts2_core::runstate::RelicRarity::Common,
            vec![RelicId::BagOfMarbles, RelicId::Gorget, RelicId::MealTicket],
        );
        rs.relic_grab_bag_by_rarity.insert(
            sts2_core::runstate::RelicRarity::Uncommon,
            vec![RelicId::Kifuda, RelicId::Shuriken, RelicId::Tingsha],
        );
        rs.relic_grab_bag_by_rarity.insert(
            sts2_core::runstate::RelicRarity::Rare,
            vec![RelicId::Brimstone, RelicId::Astrolabe, RelicId::CallingBell],
        );
        rs
    }

    #[test]
    fn ranwid_potion_trades_the_potion_he_asked_for() {
        crate::cards::dispatch::ensure_installed();
        let mut rs = ranwid_state();
        let (asked, _) = ranwid_asks_for(&rs);
        let taken = rs.potions[asked.expect("a potion on the belt")];
        let outcome = (RANWID_THE_ELDER.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.potions.len(), 1, "one potion given up");
        assert!(!rs.potions.contains(&taken));
        assert_eq!(rs.relics.len(), 4, "one relic back");
    }

    #[test]
    fn ranwid_gold_deducts_100g() {
        crate::cards::dispatch::ensure_installed();
        let mut rs = ranwid_state();
        let outcome = (RANWID_THE_ELDER.options[1].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.gold, 200 - RANWID_GOLD_COST);
        assert_eq!(rs.relics.len(), 4, "one relic back");
    }

    /// Give one relic, get TWO — the whole point of the option, and it did
    /// nothing at all before.
    #[test]
    fn ranwid_relic_takes_one_and_gives_two() {
        crate::cards::dispatch::ensure_installed();
        let mut rs = ranwid_state();
        let (_, asked) = ranwid_asks_for(&rs);
        let given = asked.expect("a tradable relic");
        let outcome = (RANWID_THE_ELDER.options[2].handler)(&mut rs);
        assert!(outcome.finished);
        assert!(!rs.relics.contains(&given), "{given:?} handed over");
        assert_eq!(rs.relics.len(), 4, "3 - 1 + 2");
    }

    /// Ranwid asks only for a TRADABLE relic, and C# `IsTradable` is false
    /// once a wax relic has melted (`RelicModel.cs:188`,
    /// `RanwidTheElder.cs:74/94`): the melted one is neither picked nor
    /// counted toward `IsAllowed`.
    #[test]
    fn ranwid_skips_melted_relics() {
        let mut rs = ranwid_state();
        rs.relics = vec![RelicId::AmethystAubergine];
        rs.wax_relics.insert(RelicId::AmethystAubergine);
        rs.melted_relics.insert(RelicId::AmethystAubergine);
        assert_eq!(ranwid_asks_for(&rs).1, None, "only relic is melted");
        assert!(!super::super::ranwid_the_elder_allowed(&rs));
        rs.melted_relics.clear();
        assert_eq!(ranwid_asks_for(&rs).1, Some(RelicId::AmethystAubergine));
        assert!(super::super::ranwid_the_elder_allowed(&rs));
    }

    // ── FakeMerchant throw_foul (mutant coverage) ────────────────────

    #[test]
    fn fake_merchant_throw_foul_stages_event_combat() {
        // Bead `sts2-rs-ck9z6` (GH #166): throw_foul now enters a real
        // event→combat (NOT a finished outcome) — it stages a no-resume
        // FakeMerchant fight whose victory reward is FakeMerchantsRug.
        let mut rs = RunState::new();
        let outcome = fake_merchant_throw_foul(&mut rs);
        assert!(!outcome.finished, "throw_foul defers finish to post-combat");
        assert!(
            !outcome.description.is_empty(),
            "throw_foul carries a description",
        );
        let req = rs.event_combat.as_ref().expect("combat staged");
        assert!(!req.resume);
        assert_eq!(
            req.encounter,
            sts2_core::events::EventEncounter::FakeMerchant
        );
    }

    // ── JungleMazeAdventure gold (mutant coverage) ───────────────────

    #[test]
    fn jungle_maze_gold_rolls_base_plus_signed_swing() {
        // Kills the `base + swing` → `base - swing` mutants AND the
        // `delete -` (on the `-15.0` swing bound) mutants. v0.107.1 MegaRandom
        // re-rolls seed 0 to a +10 solo swing and a non-zero join swing, so
        // both the sign flip and the bound-collapse still change the result.
        let rs = RunState::new();
        assert_eq!(rs.rng.seed, 0, "default seed pins the roll");
        let (solo, join) = jungle_maze_gold(&rs);
        assert_eq!(solo, 160, "150 base + (+10) solo swing");
        assert_eq!(join, 61, "50 base + (+11) join swing");
    }

    #[test]
    fn jungle_maze_generate_options_returns_two_options() {
        // Kills `-> Vec<EventOptionInfo> with vec![]`: the generator always
        // surfaces the two choices with the rolled gold interpolated.
        let rs = RunState::new();
        let opts = jungle_maze_generate_options(&rs, &JUNGLE_MAZE_ADVENTURE);
        assert_eq!(opts.len(), 2, "solo + join options");
        assert_eq!(opts[0].option_id, "solo");
        assert_eq!(opts[1].option_id, "join");
        // Rolled solo gold (160) must be interpolated into the label text.
        assert!(
            opts[0].description.contains("160"),
            "solo gold interpolated: {}",
            opts[0].description,
        );
    }

    // ── MorphicGrove group (mutant coverage) ─────────────────────────

    #[test]
    fn morphic_grove_group_auto_transforms_two_cards() {
        // Kills the `done += 1` → `*=`/`-=` mutants: with two transformable
        // deck cards the non-interactive path reports exactly 2 transformed.
        crate::cards::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.character_id = sts2_core::CharacterId::Ironclad;
        rs.gold = 150;
        rs.master_deck.clear();
        for _ in 0..2 {
            rs.master_deck.push(
                crate::cards::spec_index::card_pod_from_id(
                    sts2_core::cards::CardId::StrikeIronclad,
                    false,
                )
                .expect("strike pod exists"),
            );
        }
        assert!(!rs.interactive_card_transform, "non-interactive path");
        let outcome = morphic_grove_group(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.gold, 0, "Group loses all gold");
        assert_eq!(rs.master_deck.len(), 2, "transform replaces, not removes");
        assert!(
            outcome.description.contains("transformed 2 cards"),
            "exactly 2 transformed: {}",
            outcome.description,
        );
    }

    /// v0.107.1 (`MorphicGrove.cs:50`): the Group transform migrated from
    /// `RunState.Rng.Niche` to the event's own `base.Rng`. The non-interactive
    /// path must transform off the event stream WITHOUT advancing the run-wide
    /// niche cursor.
    #[test]
    fn morphic_grove_group_auto_transforms_off_event_stream_not_niche() {
        crate::cards::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.character_id = sts2_core::CharacterId::Ironclad;
        rs.gold = 150;
        rs.master_deck.clear();
        for _ in 0..2 {
            rs.master_deck.push(
                crate::cards::spec_index::card_pod_from_id(
                    sts2_core::cards::CardId::StrikeIronclad,
                    false,
                )
                .expect("strike pod exists"),
            );
        }
        let niche_before = rs.rng.niche.counter();
        let outcome = morphic_grove_group(&mut rs);
        assert!(outcome.finished);
        assert_eq!(
            rs.rng.niche.counter(),
            niche_before,
            "v0.107.1 transforms off the event stream, not the niche stream",
        );
    }

    /// The interactive Group path queues 2 transform picks, each tagged with
    /// the `MORPHIC_GROVE` event stream and its in-stream read index (0, 1) —
    /// so the deferred resolver rolls the SAME sequence the auto path would,
    /// matching C#'s persistent `base.Rng` across the two `TransformToRandom`
    /// calls.
    #[test]
    fn morphic_grove_group_interactive_queues_event_stream_picks() {
        let mut rs = RunState::new();
        rs.gold = 150;
        rs.interactive_card_transform = true;
        let outcome = morphic_grove_group(&mut rs);
        assert!(outcome.finished);
        assert_eq!(
            rs.pending_deck_transforms.len(),
            MORPHIC_GROVE_TRANSFORM_COUNT
        );
        assert_eq!(
            rs.pending_deck_transforms[0].event_stream_roll,
            Some((MORPHIC_GROVE_STREAM, 0)),
        );
        assert_eq!(
            rs.pending_deck_transforms[1].event_stream_roll,
            Some((MORPHIC_GROVE_STREAM, 1)),
        );
    }
}
