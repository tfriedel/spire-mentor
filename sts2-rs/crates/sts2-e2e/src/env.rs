//! `ppo.e2e.env.E2EEnv`'s decision driver at inference (default `EnvConfig`:
//! no tokenizer, no capture, no teacher): one Ironclad A10 v0.111.0 run as a
//! stream of stored decisions.
//!
//! ```rust,ignore
//! let mut env = E2eEnv::new(seed);
//! let mut s = env.reset();
//! while let Step::Decision(d) = s {
//!     s = env.step(student.choose(&d, env.rm()));   // a merged-class index
//! }
//! ```
//!
//! What it does, as the Python does it:
//!
//! * **Out of combat** (`_RunEnv._advance`): binds an event before reading
//!   its options (`materialize_event`), masks a full-belt `shop_buy_potion`
//!   (`mask_rejectable`), auto-takes a lone action, and otherwise classifies
//!   the actions (`contracts::classify_ooc`), keys them
//!   (`contracts::ooc_class_keys` over `contracts::ooc_payloads` of the
//!   decision's [`observe`]) and merges them; one merged class is applied
//!   without a decision.
//! * **In combat** (`E2EEnv._combat`): a fight is recognised by its action
//!   kinds (never by `phase()`), starts and ends with the fight's
//!   [`IntentLog`], and every step is classified on the true state and keyed
//!   on the belief view (`cs.belief_view(0)`): `merged_classes`. A lone legal
//!   action or a lone merged class is applied without a decision.
//! * **Step** (`E2EEnv.step(c)`): applies class `c`'s representative (its
//!   first member), a combat action through `combat_to_rm`.
//!
//! The IntentLog sees every combat step that can change it: it records only a
//! round's first snapshot, so a forced step later in a round skips the
//! snapshot (`FAST_PATHS`). The choose fast path (an outcome key only for
//! choose actions whose selected content another choose shares) is Python's
//! default too, and the outcome key of an unshared choose is `""`.
//!
//! Not ported (inference needs none of it): rewards, units, F9 invariant
//! sampling, practice capture, the row batch, counters.
//!
//! ## Over an existing run
//!
//! [`Driver`] is the env without the run: [`Driver::peek`] says what the env
//! does at a run position (a forced step, a decision, the end) without
//! applying anything, and [`Driver::apply`] / [`Driver::apply_with`] apply
//! one action with the env's bookkeeping. [`E2eEnv`] is `loop { peek; apply }`
//! over its own run; a host that owns the run (the workbench) peeks once per
//! position and applies what it chose, and so plays exactly the env's run.
//!
//! A driver that starts late (a run the host already moved) lacks what it
//! would have seen:
//!
//! * **the fight's [`IntentLog`]** (the snapshot's `ehist`, a policy input):
//!   rebuilt by showing the driver every combat position the run passed
//!   through since the fight started, at least the first of every round
//!   ([`Driver::observe_combat_position`], or [`Driver::set_intent_log`]).
//!   Without it the history starts empty at the current round, which only
//!   changes `ehist`.
//! * **`grant_picked`** (`Selection.picked` on a `CARD_GRANT` decision): the
//!   grant picks applied since the open grant opened
//!   ([`Driver::set_grant_picked`]; a move the host applies behind the
//!   driver's back is counted by [`Driver::note_external`]).
//! * **fight tracking** (fight sequence number, rows this fight, last round):
//!   only the per-fight decision cap and the round-reset fallback read it; a
//!   fresh driver starts the fight at the first combat position it sees.

use serde_json::{json, Value};
use smallvec::SmallVec;
use sts2_core::run_manager::{Phase, RunManager};
use sts2_core::{Action, CombatState};
use sts2_wire::emit::names::{phase_str, result_str};
use sts2_wire::emit::run_manager as emit_rm;

use crate::contracts::{self as ct, ak, dt, SchemaError, Selection};
use crate::intent_log::IntentLog;
use crate::pyjson::{self, as_int, get, get_null, list_or_empty, truthy};
use crate::snapshot;

/// The campaign build (`factory.GAME_VERSION`).
pub const GAME_VERSION: &str = "v0.111.0";
/// The campaign ascension (`factory.ASCENSION`).
pub const ASCENSION: u8 = 10;
/// All steps, forced ones included (`MAX_RUN_STEPS`): a backstop.
pub const MAX_RUN_STEPS: u64 = 200_000;

/// Per-copy card variables that change the card face (`FACE_VARS`): a play
/// key is refined by the ones a card carries. `enchantment_used`: Vigorous
/// adds its amount until the card's first play, and an Anger clone made in
/// `OnPlay` keeps it, so a spent and a fresh copy preview differently (#1097).
pub const FACE_VARS: [&str; 7] = [
    "rampage_damage",
    "claw_buff",
    "genetic_algorithm_block",
    "ritual_dagger_damage",
    "momentum_damage",
    "ethereal_this_combat",
    "enchantment_used",
];

/// The run-level forms of combat actions (`COMBAT_RUN_KINDS`).
const COMBAT_RUN_KINDS: [&str; 5] = [
    "play_card",
    "end_turn",
    "drink_potion",
    "combat_resolve_choice",
    "combat_resolve_choice_take_first",
];

/// The run-action kinds the run encoder has a surface for
/// (`encoder._KIND_SURFACE`, `assert_encodable`).
pub(crate) const ENCODABLE: [&str; 21] = [
    "move",
    "pick_card_reward",
    "sacrifice_card_reward",
    "pick_relic_reward",
    "skip_relic",
    "pick_boss_relic",
    "smith_card",
    "remove_deck_card",
    "upgrade_deck_card",
    "transform_deck_card",
    "enchant_deck_card",
    "grant_deck_card",
    "shop_buy_card",
    "shop_buy_relic",
    "shop_buy_potion",
    "shop_remove_card",
    "leave_shop",
    "rest_option",
    "event_choice",
    "collect_treasure",
    "potion_swap",
];

const DECK_MUTATION: [&str; 5] = [
    "remove_deck_card",
    "upgrade_deck_card",
    "transform_deck_card",
    "enchant_deck_card",
    "grant_deck_card",
];

pub(crate) type R<T> = Result<T, String>;

pub(crate) fn schema(e: SchemaError) -> String {
    e.to_string()
}

pub(crate) fn action_name(a: &Value) -> &str {
    get(a, "action").and_then(Value::as_str).unwrap_or("")
}

/// Install the engine's content registries (cards, relics, potions) and the
/// co-op teammate policy, as `import sts2_rs` does. Idempotent; every env
/// constructor calls it. Without it relic and potion effects silently no-op
/// on the run path (Burning Blood never heals).
pub fn ensure_engine_installed() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
    sts2_core::coop::register_teammate_policy(sts2_mcts::expert::teammate_policy);
}

/// The campaign run for `seed`: `factory.new_run` (Ironclad, A10, v0.111.0,
/// Neow start, `act1="roll"`, `mode="agent"`, interactive potion swap on),
/// built as `sts2_rs.RunManager(...)` builds it.
///
/// # Panics
///
/// If `seed` is outside `[0, 2**31)` (the factory's assert).
#[must_use]
pub fn new_run(seed: i64) -> RunManager {
    new_run_at(seed, ASCENSION)
}

/// [`new_run`] at another ascension (the workbench lets the player pick
/// one; the student only ever trained at [`ASCENSION`]).
///
/// # Panics
///
/// If `seed` is outside `[0, 2**31)`.
#[must_use]
pub fn new_run_at(seed: i64, ascension: u8) -> RunManager {
    assert!(
        (0..1_i64 << 31).contains(&seed),
        "seed {seed} outside [0, 2**31)"
    );
    ensure_engine_installed();
    let gv: sts2_core::GameVersion = GAME_VERSION.parse().expect("v0.111.0 is a known build");
    let master_seed = u64::try_from(seed).unwrap_or(0);
    let effective_seed = i32::try_from(seed).unwrap_or(0);
    let rm = sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
        effective_seed,
        sts2_core::potions::CharacterId::Ironclad,
        ascension,
    );
    finish_run(rm, gv, master_seed)
}

/// The game's seed alphabet (`SeedHelper._characters`: no `O`, no `I`).
const GAME_SEED_ALPHABET: &str = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

/// A typed game seed as the game reads it (`SeedHelper.CanonicalizeSeed`:
/// upper-cased, `O`→`0`, `I`→`1`, trimmed).
///
/// # Errors
///
/// The seed is empty, or has a character outside the game's alphabet.
pub fn canonical_game_seed(seed: &str) -> Result<String, String> {
    let s = sts2_core::canonicalize_seed(seed);
    if s.is_empty() {
        return Err("the seed is empty".to_string());
    }
    if let Some(c) = s.chars().find(|c| !GAME_SEED_ALPHABET.contains(*c)) {
        return Err(format!(
            "seed character '{c}' is not in the game's alphabet"
        ));
    }
    Ok(s)
}

/// [`new_run_at`] for a game seed string: the run the v0.111.0 game deals for
/// `seed` (canonicalised, then `RunRngSet`'s XxHash64), with the same setup.
///
/// # Errors
///
/// See [`canonical_game_seed`].
pub fn new_run_from_game_seed(seed: &str, ascension: u8) -> Result<RunManager, String> {
    let canonical = canonical_game_seed(seed)?;
    ensure_engine_installed();
    let gv: sts2_core::GameVersion = GAME_VERSION.parse().expect("v0.111.0 is a known build");
    let master_seed = gv.run_seed_hash(&canonical);
    // The i32 view `sts2_rs.RunManager(string_seed=...)` constructs with.
    #[allow(clippy::cast_possible_truncation)]
    let effective_seed = master_seed as i32;
    let rm = sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
        effective_seed,
        sts2_core::potions::CharacterId::Ironclad,
        ascension,
    );
    Ok(finish_run(rm, gv, master_seed))
}

/// The campaign's run setup shared by [`new_run_at`] and
/// [`new_run_from_game_seed`].
fn finish_run(rm: RunManager, gv: sts2_core::GameVersion, master_seed: u64) -> RunManager {
    let mut rm = rm
        .with_game_version(gv)
        .with_run_seed(master_seed)
        .with_neow_start();
    rm.run_state.interactive_card_removal = false;
    rm.run_state.interactive_card_upgrade = false;
    rm.run_state.interactive_card_transform = false;
    rm.run_state.interactive_potion_swap = true;
    rm.run_state.direct_event_relic_grant = false;
    rm.run_state.set_agent_decisions();
    let mut rm = rm.with_rolled_act1();
    sts2_content::run_callbacks::initialize_run_setup(&mut rm);
    rm
}

/// Whether a run-level action list is a fight's (`is_combat`: by kind).
#[must_use]
/// The player's current HP: the fight's while one is on, else the run's.
pub fn hp_now(rm: &RunManager) -> i64 {
    rm.combat.as_ref().map_or_else(
        || i64::from(rm.run_state.current_hp),
        |c| i64::from(c.player.current_hp),
    )
}

pub fn is_combat(actions: &[Value]) -> bool {
    !actions.is_empty()
        && actions
            .iter()
            .all(|a| COMBAT_RUN_KINDS.contains(&action_name(a)))
}

/// `ppo.env.classify_surface(phase, actions)`.
#[must_use]
pub fn classify_surface(phase: &str, actions: &[Value]) -> Option<&'static str> {
    if actions
        .first()
        .is_some_and(|a| action_name(a) == "smith_card")
    {
        return Some("card-pick");
    }
    if !actions.is_empty()
        && actions
            .iter()
            .all(|a| DECK_MUTATION.contains(&action_name(a)))
    {
        return Some("card-pick");
    }
    if !actions.is_empty() && actions.iter().all(|a| action_name(a) == "potion_swap") {
        return Some("potion-swap");
    }
    match phase {
        "map_choice" => Some("path"),
        "card_reward" | "boss_relic" => Some("card-pick"),
        "event" => Some("event"),
        "shop" => Some("shop"),
        "rest_site" => Some("rest"),
        "treasure" => Some("treasure"),
        _ => None,
    }
}

/// `ppo.env.mask_rejectable`: in a shop with a full belt, drop
/// `shop_buy_potion` (a graceful no-op the engine accepts).
#[must_use]
pub fn mask_rejectable(rm: &RunManager, phase: &str, actions: Vec<Value>) -> Vec<Value> {
    if phase != "shop" {
        return actions;
    }
    let held = emit_rm::potions(rm).len();
    if i64::try_from(held).unwrap_or(i64::MAX) < i64::from(rm.run_state.max_potion_slots) {
        return actions;
    }
    let masked: Vec<Value> = actions
        .iter()
        .filter(|a| action_name(a) != "shop_buy_potion")
        .cloned()
        .collect();
    if masked.is_empty() {
        actions
    } else {
        masked
    }
}

fn ench_list(v: Vec<Option<(String, i64)>>) -> Value {
    json!(v)
}

/// `ppo.env.observe(rm, phase, surface, seed, ascension)`: the run
/// observation (`run_obs`), as Python's dict serializes. Binds the event on an
/// event surface (idempotent).
pub fn observe(
    rm: &mut RunManager,
    phase: &str,
    surface: Option<&str>,
    seed: i64,
    ascension: i64,
) -> Value {
    let deck = emit_rm::master_deck_detailed(rm);
    let mut o = pyjson::obj();
    o.insert("phase".into(), json!(phase));
    o.insert("surface".into(), json!(surface));
    o.insert("seed".into(), json!(seed));
    o.insert("ascension".into(), json!(ascension));
    o.insert("deck".into(), json!(deck));
    o.insert(
        "deck_enchantments".into(),
        ench_list(emit_rm::master_deck_enchantments(rm)),
    );
    o.insert("relics".into(), json!(emit_rm::relics(rm)));
    o.insert("potions".into(), json!(emit_rm::potions(rm)));
    o.insert("map".into(), emit_rm::full_map(rm));
    o.insert("map_coord".into(), json!(emit_rm::map_coord(rm)));
    // metrics_snapshot
    o.insert("ascension".into(), json!(rm.run_state.ascension_level));
    o.insert("total_floor".into(), json!(rm.run_state.total_floor));
    o.insert("act_index".into(), json!(rm.run_state.act_index));
    o.insert("act_floor".into(), json!(rm.run_state.act_floor));
    o.insert("hp".into(), json!(rm.run_state.current_hp));
    o.insert("max_hp".into(), json!(rm.run_state.max_hp));
    o.insert("gold".into(), json!(emit_rm::gold(rm)));
    o.insert("deck_size".into(), json!(deck.len()));
    o.insert(
        "num_upgraded".into(),
        json!(deck.iter().filter(|(_, u)| *u).count()),
    );
    o.insert("num_relics".into(), json!(emit_rm::relics(rm).len()));
    o.insert("num_potions".into(), json!(emit_rm::potions(rm).len()));
    match surface {
        Some("card-pick") => {
            let upgradable: Vec<usize> = rm
                .run_state
                .master_deck
                .iter()
                .enumerate()
                .filter(|(_, c)| c.is_upgradable())
                .map(|(i, _)| i)
                .collect();
            o.insert(
                "card_reward_option_ids".into(),
                json!(emit_rm::card_reward_option_ids(rm)),
            );
            o.insert(
                "card_reward_option_upgraded".into(),
                json!(emit_rm::card_reward_option_upgraded(rm)),
            );
            o.insert(
                "card_reward_option_enchantments".into(),
                ench_list(emit_rm::card_reward_option_enchantments(rm)),
            );
            o.insert(
                "card_reward_bundles".into(),
                json!(emit_rm::card_reward_bundles(rm)),
            );
            o.insert(
                "relic_reward_ids".into(),
                json!(emit_rm::relic_reward_ids(rm)),
            );
            o.insert(
                "boss_relic_options".into(),
                json!(emit_rm::boss_relic_options(rm)),
            );
            o.insert("smith_options".into(), json!(emit_rm::smith_options(rm)));
            o.insert(
                "smith_option_enchantments".into(),
                ench_list(emit_rm::smith_option_enchantments(rm)),
            );
            o.insert(
                "card_removal_options".into(),
                json!(emit_rm::card_removal_options(rm)),
            );
            o.insert(
                "card_removal_option_enchantments".into(),
                ench_list(emit_rm::card_removal_option_enchantments(rm)),
            );
            o.insert(
                "card_grant_options".into(),
                json!(emit_rm::card_grant_options(rm)),
            );
            o.insert(
                "card_grant_option_enchantments".into(),
                ench_list(emit_rm::card_grant_option_enchantments(rm)),
            );
            o.insert(
                "transform_select_deck_indices".into(),
                json!(rm.transform_eligible_deck_indices()),
            );
            o.insert(
                "enchant_select_deck_indices".into(),
                json!(rm.enchant_eligible_deck_indices()),
            );
            o.insert(
                "pending_enchantment".into(),
                json!(rm
                    .run_state
                    .pending_deck_enchants
                    .first()
                    .map(|p| p.enchantment.python_name())),
            );
            o.insert("upgrade_select_deck_indices".into(), json!(upgradable));
        }
        Some("shop") => {
            o.insert("shop".into(), emit_rm::shop_inventory(rm));
        }
        Some("event") => {
            sts2_content::events::dispatch::bind_pending_event_callback(&mut rm.run_state);
            o.insert("event_name".into(), json!(current_event_name(rm)));
            o.insert("event_options".into(), emit_rm::event_option_infos(rm));
        }
        Some("rest") => {
            // `ppo.env.observe`: the engine's current option ids.
            let ids: Vec<&str> = rm
                .run_state
                .rest_site_options()
                .iter()
                .map(sts2_core::runstate::RestSiteOption::option_id)
                .collect();
            o.insert("rest_option_labels".into(), json!(ids));
        }
        Some("potion-swap") => {
            o.insert(
                "pending_potion_swap".into(),
                json!(rm
                    .run_state
                    .pending_potion_swaps
                    .first()
                    .map(|k| format!("{k:?}"))),
            );
        }
        _ => {}
    }
    Value::Object(o)
}

/// `rm.current_event_name()`.
#[must_use]
pub fn current_event_name(rm: &RunManager) -> Option<String> {
    rm.run_state
        .current_event
        .map(|id| id.python_name().to_string())
}

/// `fight_room(rm)`: `monster`, `elite` or `boss` (anything else reads as
/// `monster`).
#[must_use]
pub fn fight_room(rm: &RunManager) -> &'static str {
    match get(&emit_rm::current_room(rm), "room_type").and_then(Value::as_str) {
        Some("elite") => "elite",
        Some("boss") => "boss",
        _ => "monster",
    }
}

/// `combat_bench.oracle.combat_to_rm`: a `legal_actions` dict as the
/// `RunManager.take_action` dict.
///
/// # Errors
///
/// A kind with no run-action form.
pub fn combat_to_rm(a: &Value) -> R<Value> {
    Ok(match action_name(a) {
        "play" => json!({
            "action": "play_card",
            "hand_idx": get_null(a, "card_index"),
            "target_index": get_null(a, "target_index"),
        }),
        "drink_potion" | "potion" => json!({
            "action": "drink_potion",
            "potion_slot": get_null(a, "slot"),
            "target_index": get_null(a, "target_index"),
        }),
        "choose" => {
            let sel = list_or_empty(a, "selection").to_vec();
            if sel.len() <= 1 {
                json!({"action": "combat_resolve_choice", "option": sel.first().cloned()})
            } else {
                json!({"action": "combat_resolve_choice_selection", "selection": sel})
            }
        }
        "end_turn" => json!({"action": "end_turn"}),
        other => return Err(format!("no run-action form for combat action {other:?}")),
    })
}

/// `RunManager.take_action(dict)`, with the explicit multi-pick
/// `combat_resolve_choice_selection` routed as the `PyO3` binding routes it.
/// Returns the result string.
///
/// # Errors
///
/// An action dict the parser refuses.
pub fn take_run_action(rm: &mut RunManager, a: &Value) -> R<&'static str> {
    if action_name(a) == "combat_resolve_choice_selection" {
        let sel: SmallVec<[u8; 4]> = list_or_empty(a, "selection")
            .iter()
            .map(|x| {
                as_int(x)
                    .and_then(|i| u8::try_from(i).ok())
                    .ok_or_else(|| format!("bad selection entry {x}"))
            })
            .collect::<R<_>>()?;
        return Ok(result_str(&rm.resolve_combat_choice_selection(sel)));
    }
    let ra = sts2_wire::emit::run_action_from_value(a).map_err(|e| e.to_string())?;
    Ok(result_str(&rm.take_action(ra)))
}

/// `parse_bridge_action`: a `legal_actions`-shape dict as a core [`Action`]
/// (`CombatState.apply`'s decoder, solo seat).
///
/// # Errors
///
/// An unknown kind or a missing / out-of-range index.
pub fn bridge_action(a: &Value) -> R<Action> {
    let u8_of = |k: &str| -> R<Option<u8>> {
        match get(a, k) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => as_int(v)
                .and_then(|i| u8::try_from(i).ok())
                .map(Some)
                .ok_or_else(|| format!("apply: '{k}' must be a u8, got {v}")),
        }
    };
    let req = |k: &str| -> R<u8> { u8_of(k)?.ok_or_else(|| format!("apply: missing '{k}'")) };
    if u8_of("ally_index")?.is_some() {
        return Err("apply: co-op ally actions are not supported".to_owned());
    }
    Ok(match action_name(a) {
        "play" => Action::PlayCard {
            hand_index: req("card_index")?,
            target: u8_of("target_index")?,
        },
        "end_turn" => Action::EndTurn,
        "potion" | "drink_potion" => Action::DrinkPotion {
            potion_slot: req("slot")?,
            target: u8_of("target_index")?,
        },
        "choose" => {
            let mut sel: SmallVec<[u8; 4]> = SmallVec::new();
            match get(a, "selection") {
                Some(Value::Array(xs)) => {
                    for x in xs {
                        sel.push(
                            as_int(x)
                                .and_then(|i| u8::try_from(i).ok())
                                .ok_or_else(|| format!("apply: bad selection entry {x}"))?,
                        );
                    }
                }
                _ => sel.push(req("index")?),
            }
            Action::ResolveChoice { selection: sel }
        }
        other => return Err(format!("apply: unknown action {other:?}")),
    })
}

fn face_vars(card: &Value) -> serde_json::Map<String, Value> {
    let cv = get_null(card, "combat_vars");
    let mut out = pyjson::obj();
    for k in FACE_VARS {
        if let Some(v) = get(&cv, k).filter(|v| truthy(v)) {
            out.insert(k.to_owned(), v.clone());
        }
    }
    out
}

/// One combat state classified and deduped as the env does
/// (`merged_classes`).
#[derive(Clone, Debug)]
pub struct CombatClasses {
    /// `COMBAT_TURN` or `COMBAT_CHOICE`.
    pub dtype: u8,
    /// Action kind per member.
    pub kinds: Vec<u8>,
    /// The members in the belief view's index space.
    pub obs_actions: Vec<Value>,
    /// Merged class per member.
    pub classes: Vec<usize>,
    /// Member indices per class, in order.
    pub members: Vec<Vec<usize>>,
    /// Merge key per member.
    pub keys: Vec<String>,
}

/// `combat_keys(bv, d_bv, acts)`: `contracts::combat_class_keys` with each
/// play key refined by the card's [`FACE_VARS`], and the choose fast path
/// (the outcome key only for a choose whose selection another one shares).
///
/// # Errors
///
/// A schema error, or an action the belief view cannot apply.
pub fn combat_keys(bv: &CombatState, d_bv: &Value, acts: &[Value]) -> R<Vec<String>> {
    let deck = get_null(d_bv, "deck");
    let hand = list_or_empty(&deck, "hand");
    let player = get_null(d_bv, "player");
    let pots = list_or_empty(&player, "potions");
    let choice = sts2_wire::emit::pending_choice(bv);
    let shared: Option<std::collections::HashSet<usize>> = match choice.as_ref() {
        Some(ch) if truthy(ch) => {
            let mut sel: Vec<(usize, Vec<String>)> = Vec::new();
            for (i, a) in acts.iter().enumerate() {
                if action_name(a) == "choose" {
                    sel.push((i, ct::selected_keys(a, ch, &deck).map_err(schema)?));
                }
            }
            let mut n: std::collections::BTreeMap<&Vec<String>, usize> =
                std::collections::BTreeMap::new();
            for (_, k) in &sel {
                *n.entry(k).or_default() += 1;
            }
            Some(
                sel.iter()
                    .filter(|(_, k)| n[k] > 1)
                    .map(|(i, _)| *i)
                    .collect(),
            )
        }
        _ => None,
    };
    let mut outcome = |i: usize| -> Result<String, SchemaError> {
        if shared.as_ref().is_some_and(|s| !s.contains(&i)) {
            return Ok(String::new());
        }
        let mut c = bv.clone();
        let act = bridge_action(&acts[i]).map_err(SchemaError)?;
        let _ = sts2_core::apply(&mut c, act);
        snapshot::choose_outcome_key(&c).map_err(SchemaError)
    };
    let keys = ct::combat_class_keys(acts, hand, pots, choice.as_ref(), &mut outcome, &deck)
        .map_err(schema)?;
    let mut out = Vec::with_capacity(keys.len());
    for (k, a) in keys.into_iter().zip(acts) {
        if action_name(a) == "play" {
            let fv = get(a, "card_index")
                .and_then(as_int)
                .and_then(|i| usize::try_from(i).ok())
                .and_then(|i| hand.get(i))
                .map(face_vars)
                .unwrap_or_default();
            if !fv.is_empty() {
                out.push(pyjson::dumps(&json!([k, fv])));
                continue;
            }
        }
        out.push(k);
    }
    Ok(out)
}

/// `merged_classes(cs, legal, view=(bv, d_bv, _))`: classify the true state's
/// legal actions and dedupe them on the belief view.
///
/// # Errors
///
/// A schema error or an unmappable choice.
pub fn merged_classes(
    cs: &CombatState,
    legal: &[Value],
    bv: &CombatState,
    d_bv: &Value,
) -> R<CombatClasses> {
    let cs_choice = sts2_wire::emit::pending_choice(cs);
    let bv_choice = sts2_wire::emit::pending_choice(bv);
    let (dtype, kinds) = ct::classify_combat(legal, cs_choice.is_some()).map_err(schema)?;
    let acts = snapshot::obs_actions(legal, cs_choice.as_ref(), bv_choice.as_ref())?;
    let keys = combat_keys(bv, d_bv, &acts)?;
    let (classes, members) = ct::classes_from_keys(&keys);
    Ok(CombatClasses {
        dtype,
        kinds,
        obs_actions: acts,
        classes,
        members,
        keys,
    })
}

/// Whether a stored decision is a combat or an out-of-combat one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionKind {
    /// A fight's decision (`COMBAT_TURN`, `COMBAT_CHOICE`).
    Combat,
    /// An out-of-combat decision.
    Ooc,
}

impl DecisionKind {
    /// `"combat"` / `"ooc"`, as the fixtures spell it.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Combat => "combat",
            Self::Ooc => "ooc",
        }
    }
}

/// A combat decision's view: what `tokens.encode_combat` reads.
#[derive(Clone, Debug)]
pub struct CombatView {
    /// `cs.belief_view(0)`: the honest state the policy may read.
    pub bv: CombatState,
    /// `state_to_dict(bv)`.
    pub d_bv: Value,
    /// The honest snapshot with this fight's `ehist`.
    pub snap: Value,
    /// `monster` / `elite` / `boss`.
    pub room: &'static str,
    /// `bv.encounter_id` (RUNCTX `X_ID_AUX`).
    pub encounter_id: Option<&'static str>,
    /// Fight sequence number in the segment.
    pub fight: usize,
    /// The fight's round counter.
    pub turn: i64,
    /// Stored rows of this fight before this one.
    pub fight_row: usize,
}

/// An out-of-combat decision's view.
#[derive(Clone, Debug)]
pub struct OocView {
    /// `rm.phase()`.
    pub phase: &'static str,
    /// `classify_surface(phase, actions)`.
    pub surface: Option<&'static str>,
    /// `ppo.env.observe(...)` (the tokenizer reads `honest_obs` of it).
    pub run_obs: Value,
}

/// One stored decision (`E2EDecision`): at least two merged classes.
///
/// The student answers with a class index into `members`; the env applies
/// the class's first member. The live run is [`E2eEnv::rm`] (read it, never
/// mutate it, until the next step).
#[derive(Clone, Debug)]
pub struct Decision {
    /// Index of this stored row in the run.
    pub row: usize,
    /// Combat or out of combat.
    pub kind: DecisionKind,
    /// `contracts::dt` id.
    pub dtype: u8,
    /// Action kind per member.
    pub kinds: Vec<u8>,
    /// Combat: `cs.legal_actions()` (true state); OOC: the run actions.
    pub actions: Vec<Value>,
    /// Combat: the members in the belief view's index space; OOC: `actions`.
    pub obs_actions: Vec<Value>,
    /// Merged class per member (`member_class`).
    pub classes: Vec<usize>,
    /// Member indices per class, in member order.
    pub members: Vec<Vec<usize>>,
    /// Merge key per member.
    pub keys: Vec<String>,
    /// The open selection (`COMBAT_CHOICE`, `CARD_GRANT`).
    pub sel: Selection,
    /// `rm.current_event_name()` on `EVENT` and `NEOW`.
    pub event_name: Option<String>,
    /// `rm.act_index`.
    pub act: u8,
    /// `rm.total_floor()`.
    pub floor: i32,
    /// The combat view (combat decisions).
    pub combat: Option<CombatView>,
    /// The run observation (OOC decisions).
    pub ooc: Option<OocView>,
}

impl Decision {
    /// Number of merged classes (the policy's options).
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.members.len()
    }

    /// Whether this is a combat decision.
    #[must_use]
    pub fn is_combat(&self) -> bool {
        self.kind == DecisionKind::Combat
    }

    /// The action kind of each class (its representative's): `cls_kind`.
    #[must_use]
    pub fn cls_kind(&self) -> Vec<u8> {
        self.members.iter().map(|m| self.kinds[m[0]]).collect()
    }

    /// The fight's round (combat), else `None` (the fixture's `turn`).
    #[must_use]
    pub fn turn(&self) -> Option<i64> {
        self.combat.as_ref().map(|c| c.turn)
    }
}

/// The end of a run.
#[derive(Clone, Debug)]
pub struct Terminal {
    /// `won`, `lost`, `capped` or `error`.
    pub outcome: &'static str,
    /// The run was won.
    pub won: bool,
    /// The cap that ended a capped run (`steps`, `turns`, `fight_decisions`,
    /// `run_decisions`).
    pub cap: Option<&'static str>,
    /// What went wrong, for an `error` run.
    pub error: Option<String>,
    /// Stored decisions.
    pub rows: usize,
    /// All steps, forced ones included.
    pub steps: u64,
    /// `rm.total_floor()` at the end.
    pub floor: i32,
    /// `rm.act_index` at the end.
    pub act: u8,
}

/// What [`E2eEnv::reset`] and [`E2eEnv::step`] return.
#[derive(Clone, Debug)]
pub enum Step {
    /// A stored decision: answer with [`E2eEnv::step`].
    Decision(Box<Decision>),
    /// The run is over.
    Terminal(Terminal),
}

/// How an action the env takes is applied (and checked).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyKind {
    /// An out-of-combat run action. `strict`: an engine rejection
    /// (`shop_purchase_failed`, `shop_remove_card_failed`,
    /// `invalid_for_phase`) is an error (`_RunEnv.step`); the run stepper's
    /// own lone-action auto-take is not strict (`_advance`).
    Run {
        /// Treat an engine rejection as an error.
        strict: bool,
    },
    /// A combat action in its run-action form (`_combat_apply`).
    Combat,
}

/// A step the env takes without asking the student: apply
/// [`Forced::action`] with [`Driver::apply`] and peek again.
#[derive(Clone, Debug)]
pub struct Forced {
    /// The run-action dict (`RunManager.take_action`'s; a combat action is
    /// already in its run form, [`combat_to_rm`]).
    pub action: Value,
    /// How to apply it.
    pub kind: ApplyKind,
}

/// What the env does at the current run position ([`Driver::peek`]).
#[derive(Clone, Debug)]
pub enum Next {
    /// A forced step (a lone legal action, a lone merged class).
    Forced(Forced),
    /// A stored decision: answer with one of its classes and apply the
    /// class's representative ([`Decision::class_action`]).
    Decision(Box<Decision>),
    /// The run is over (or capped / errored).
    Terminal(Terminal),
}

impl Decision {
    /// The run-action dict and its [`ApplyKind`] for merged class `class`
    /// (its first member), as [`E2eEnv::step`] applies it.
    ///
    /// # Errors
    ///
    /// `class` out of range, or a combat action with no run-action form.
    pub fn class_action(&self, class: usize) -> R<(Value, ApplyKind)> {
        let ai = *self
            .members
            .get(class)
            .and_then(|m| m.first())
            .ok_or_else(|| format!("class {class} out of range ({})", self.members.len()))?;
        self.member_action(ai)
    }

    /// The run-action dict and its [`ApplyKind`] for member `ai`.
    ///
    /// # Errors
    ///
    /// `ai` out of range, or a combat action with no run-action form.
    pub fn member_action(&self, ai: usize) -> R<(Value, ApplyKind)> {
        let a = self
            .actions
            .get(ai)
            .ok_or_else(|| format!("action {ai} out of range"))?;
        if self.is_combat() {
            Ok((combat_to_rm(a)?, ApplyKind::Combat))
        } else {
            Ok((a.clone(), ApplyKind::Run { strict: true }))
        }
    }
}

/// The open card grant as the driver sees it: `(options, pick)`, for
/// [`Driver::note_external`].
pub type GrantMark = Option<(usize, usize)>;

/// The live fight (`_Fight`).
#[derive(Clone, Debug)]
struct Fight {
    seq: usize,
    room: &'static str,
    at: (u8, Option<(i32, i32)>),
    ilog: IntentLog,
    round: i64,
    rows: usize,
}

/// `E2EEnv`'s driver state without the run: everything the env carries
/// besides the `RunManager` (fight tracking and its [`IntentLog`],
/// `grant_picked`, step / row counters, the terminal).
///
/// The env is `loop { peek; apply }`: [`Driver::peek`] looks at the run's
/// current position and says what the env does there — a [`Next::Forced`]
/// step, a [`Next::Decision`] or the [`Next::Terminal`] — WITHOUT applying
/// anything; [`Driver::apply`] (or [`Driver::apply_with`], when the caller
/// owns how actions reach its run) applies one action with the env's
/// bookkeeping. [`E2eEnv`] is exactly that loop over its own run, so a host
/// that owns the run (the workbench) reproduces the env by peeking once per
/// position and applying what it chose.
///
/// `peek` is not idempotent at a decision (it counts the row, `fight_rows`
/// / `MAX_RUN_DECISIONS` caps read the counts): ask "what would the student
/// do here?" on a clone.
#[derive(Clone, Debug)]
pub struct Driver {
    seed: i64,
    steps: u64,
    max_steps: u64,
    grant_picked: i64,
    fight: Option<Fight>,
    fights: usize,
    rows: usize,
    terminal: Option<Terminal>,
}

impl Driver {
    /// A fresh driver (a run start, or the first look at a run).
    #[must_use]
    pub fn new(seed: i64) -> Self {
        ensure_engine_installed();
        Self {
            seed,
            steps: 0,
            max_steps: MAX_RUN_STEPS,
            grant_picked: 0,
            fight: None,
            fights: 0,
            rows: 0,
            terminal: None,
        }
    }

    /// Stored decisions so far.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// All steps so far, forced ones included.
    #[must_use]
    pub const fn steps(&self) -> u64 {
        self.steps
    }

    /// The terminal, once the run has ended.
    #[must_use]
    pub const fn terminal(&self) -> Option<&Terminal> {
        self.terminal.as_ref()
    }

    /// Grant picks applied since the open card grant opened.
    #[must_use]
    pub const fn grant_picked(&self) -> i64 {
        self.grant_picked
    }

    /// Set the grant picks applied since the open grant opened.
    pub fn set_grant_picked(&mut self, n: i64) {
        self.grant_picked = n;
    }

    /// The live fight's intent log, if a fight is tracked.
    #[must_use]
    pub fn intent_log(&self) -> Option<&IntentLog> {
        self.fight.as_ref().map(|f| &f.ilog)
    }

    /// Replace the live fight's intent log (starting the fight if the run
    /// is in one and none is tracked).
    pub fn set_intent_log(&mut self, rm: &RunManager, ilog: IntentLog) {
        if self.fight.is_none() && in_combat(rm) {
            self.start_fight(rm);
        }
        if let Some(f) = self.fight.as_mut() {
            f.ilog = ilog;
        }
    }

    /// Show the driver the current combat position (after an action the
    /// caller applied to the run behind its back): starts or rolls the fight
    /// as the env would and feeds its [`IntentLog`]. A no-op outside a fight.
    ///
    /// # Errors
    ///
    /// The combat state could not be emitted.
    pub fn observe_combat_position(&mut self, rm: &RunManager) -> R<()> {
        if !in_combat(rm) {
            if self.fight.is_some() && rm.phase() != Phase::Combat {
                self.fight = None;
            }
            return Ok(());
        }
        self.track_fight(rm);
        let cs = rm.combat.as_ref().ok_or("no combat state")?;
        let bv = snapshot::belief_view(cs);
        let d_bv = snapshot::state_dict(&bv)?;
        let rnd = round_of(&d_bv);
        let f = self.roll_fight(rm, rnd);
        f.round = rnd;
        let room = f.room;
        let snap = snapshot_of(&bv, &d_bv, room);
        if let Some(f) = self.fight.as_mut() {
            f.ilog.update(&snap);
        }
        Ok(())
    }

    /// What the env does at `rm`'s current position (see the type docs).
    /// Binds a pending event's callback (idempotent), nothing else touches
    /// the run.
    pub fn peek(&mut self, rm: &mut RunManager) -> Next {
        if let Some(t) = self.terminal.as_ref() {
            return Next::Terminal(t.clone());
        }
        match self.peek_inner(rm) {
            Ok(n) => n,
            Err(e) => Next::Terminal(self.error(rm, e)),
        }
    }

    /// Apply `action` (a run-action dict) to `rm` with the env's
    /// bookkeeping. Returns the engine's result string.
    ///
    /// # Errors
    ///
    /// An action the parser refuses, or (strict) one the engine rejects.
    /// The error is NOT recorded as the terminal (the caller decides).
    pub fn apply(
        &mut self,
        rm: &mut RunManager,
        action: &Value,
        kind: ApplyKind,
    ) -> R<&'static str> {
        self.apply_with(rm, action, kind, take_run_action)
    }

    /// [`Self::apply`] with the caller's own way of putting `action` on the
    /// run (`take`: the run and the dict → the result string), for a host
    /// that wraps `take_action` (display logs and the like).
    ///
    /// # Errors
    ///
    /// Whatever `take` returns, or (strict) an engine rejection.
    pub fn apply_with<F>(
        &mut self,
        rm: &mut RunManager,
        action: &Value,
        kind: ApplyKind,
        take: F,
    ) -> R<&'static str>
    where
        F: FnOnce(&mut RunManager, &Value) -> R<&'static str>,
    {
        match kind {
            ApplyKind::Combat => {
                let res = take(rm, action)?;
                self.steps += 1;
                if rm.is_over() || rm.phase() != Phase::Combat {
                    self.fight = None;
                }
                Ok(res)
            }
            ApplyKind::Run { strict } => {
                self.note_grant(rm, action);
                let res = take(rm, action)?;
                if strict
                    && matches!(
                        res,
                        "shop_purchase_failed" | "shop_remove_card_failed" | "invalid_for_phase"
                    )
                {
                    return Err(format!("engine rejected action {action}: {res}"));
                }
                self.steps += 1;
                Ok(res)
            }
        }
    }

    /// The open grant before an action the driver does not see as a dict
    /// (a host's own verb): pass it to [`Self::note_external`] after the
    /// action.
    #[must_use]
    pub fn grant_mark(rm: &RunManager) -> GrantMark {
        rm.run_state
            .pending_card_grants
            .first()
            .map(|g| (g.options.len(), g.pick))
    }

    /// The bookkeeping of an action applied behind the driver's back (a
    /// human's move in the workbench): one step; out of combat,
    /// `grant_picked` counts the action as a grant pick iff a grant was open
    /// before (`before`, [`Self::grant_mark`]) and it closed or lost a pick
    /// (a confirm that closes a grant counts too — harmless, the next action
    /// resets it); in combat, the fight ends as [`ApplyKind::Combat`] ends
    /// it.
    pub fn note_external(&mut self, rm: &RunManager, before: GrantMark, combat: bool) {
        self.steps += 1;
        if combat {
            if rm.is_over() || rm.phase() != Phase::Combat {
                self.fight = None;
            }
            return;
        }
        let picked = match (before, Self::grant_mark(rm)) {
            (Some(_), None) => true,
            (Some((_, p0)), Some((_, p1))) => p1 < p0,
            _ => false,
        };
        if picked {
            self.grant_picked += 1;
        } else {
            self.grant_picked = 0;
        }
    }

    /// `_RunEnv.take`'s `grant_picked` bookkeeping, before the action.
    fn note_grant(&mut self, rm: &RunManager, action: &Value) {
        if action_name(action) == "grant_deck_card" {
            let idx = get(action, "idx")
                .filter(|v| truthy(v))
                .and_then(as_int)
                .unwrap_or(0);
            let n = i64::try_from(emit_rm::card_grant_options(rm).len()).unwrap_or(0);
            if idx < n {
                self.grant_picked += 1;
            }
        } else {
            self.grant_picked = 0;
        }
    }

    // ── one look (`_advance` + `_ooc` / `_combat`, minus the applies) ───

    fn peek_inner(&mut self, rm: &mut RunManager) -> R<Next> {
        if rm.is_over() {
            return Ok(self.finish(rm, "lost"));
        }
        if self.steps >= self.max_steps {
            return Ok(self.finish(rm, "max_steps"));
        }
        let phase = phase_str(rm.phase());
        if phase == "event" {
            sts2_content::events::dispatch::bind_pending_event_callback(&mut rm.run_state);
        }
        let actions = available(rm);
        if actions.is_empty() {
            return Ok(self.finish(rm, "no_actions"));
        }
        let actions = mask_rejectable(rm, phase, actions);
        if is_combat(&actions) {
            return self.combat(rm);
        }
        if self.fight.is_some() && rm.phase() != Phase::Combat {
            self.fight = None;
        }
        if let Some(a) = claws_auto_pick(rm, &actions) {
            return Ok(Next::Forced(Forced {
                action: a,
                kind: ApplyKind::Run { strict: false },
            }));
        }
        if let Some(a) = actions
            .iter()
            .find(|a| !ENCODABLE.contains(&action_name(a)))
        {
            return Err(format!(
                "unencodable action kind {:?} (phase {phase:?})",
                action_name(a)
            ));
        }
        let surface = classify_surface(phase, &actions);
        if actions.len() == 1 {
            return Ok(Next::Forced(Forced {
                action: actions[0].clone(),
                kind: ApplyKind::Run { strict: false },
            }));
        }
        let asc = i64::from(rm.run_state.ascension_level);
        let obs = observe(rm, phase, surface, self.seed, asc);
        self.ooc(rm, phase, surface, actions, obs)
    }

    /// `_ooc`: a stored OOC decision, or the lone merged class as a forced
    /// step.
    fn ooc(
        &mut self,
        rm: &RunManager,
        phase: &'static str,
        surface: Option<&'static str>,
        actions: Vec<Value>,
        obs: Value,
    ) -> R<Next> {
        let has = |n: &str| actions.iter().any(|a| action_name(a) == n);
        let event = if has("event_choice") {
            current_event_name(rm)
        } else {
            None
        };
        let grant_n = has("grant_deck_card").then(|| emit_rm::card_grant_options(rm).len());
        let bundles =
            get(&obs, "card_reward_bundles").is_some_and(truthy) && has("pick_card_reward");
        let (dtype, kinds) =
            ct::classify_ooc(&actions, event.as_deref(), bundles, grant_n).map_err(schema)?;
        let payloads = ct::ooc_payloads(&obs, &actions, &kinds).map_err(schema)?;
        let keys = ct::ooc_class_keys(&kinds, &payloads).map_err(schema)?;
        let (classes, members) = ct::classes_from_keys(&keys);
        if members.len() == 1 {
            return Ok(Next::Forced(Forced {
                action: actions[members[0][0]].clone(),
                kind: ApplyKind::Run { strict: true },
            }));
        }
        check_caps(actions.len(), members.len(), dtype)?;
        let sel = if dtype == dt::CARD_GRANT {
            Selection {
                min: rm
                    .run_state
                    .pending_card_grants
                    .first()
                    .map_or(0, |g| i64::try_from(g.min_pick).unwrap_or(i64::MAX)),
                max: i64::try_from(emit_rm::card_grant_options(rm).len()).unwrap_or(0),
                picked: self.grant_picked,
                can_skip: kinds.contains(&ak::GRANT_CONFIRM),
            }
        } else {
            Selection::default()
        };
        let event_name = if matches!(dtype, dt::EVENT | dt::NEOW) {
            current_event_name(rm)
        } else {
            None
        };
        if self.rows >= ct::MAX_RUN_DECISIONS {
            return Ok(Next::Terminal(self.cap(rm, "run_decisions")));
        }
        let d = Decision {
            row: self.rows,
            kind: DecisionKind::Ooc,
            dtype,
            kinds,
            obs_actions: actions.clone(),
            actions,
            classes,
            members,
            keys,
            sel,
            event_name,
            act: rm.run_state.act_index,
            floor: rm.run_state.total_floor,
            combat: None,
            ooc: Some(OocView {
                phase,
                surface,
                run_obs: obs,
            }),
        };
        self.rows += 1;
        Ok(Next::Decision(Box::new(d)))
    }

    /// `_combat`: one look at a fight position — a forced step, a stored
    /// decision, or a cap.
    fn combat(&mut self, rm: &RunManager) -> R<Next> {
        if self.steps >= self.max_steps {
            return Ok(Next::Terminal(self.cap(rm, "steps")));
        }
        let cs = rm
            .combat
            .as_ref()
            .ok_or("the run offers combat actions but has no combat state")?;
        let legal = match sts2_wire::emit::legal_actions(cs) {
            Value::Array(a) => a,
            _ => Vec::new(),
        };
        if legal.is_empty() {
            return Err("the run offers combat actions but the fight has none".to_owned());
        }
        self.track_fight(rm);
        let bv = snapshot::belief_view(cs);
        let d_bv = snapshot::state_dict(&bv)?;
        let rnd = round_of(&d_bv);
        self.roll_fight(rm, rnd).round = rnd;
        if rnd > ct::MAX_FIGHT_TURNS {
            return Ok(Next::Terminal(self.cap(rm, "turns")));
        }
        let forced = |a: &Value| -> R<Next> {
            Ok(Next::Forced(Forced {
                action: combat_to_rm(a)?,
                kind: ApplyKind::Combat,
            }))
        };
        let f = self.fight.as_mut().ok_or("no fight")?;
        if legal.len() == 1 && f.ilog.last_round == Some(rnd) {
            return forced(&legal[0]);
        }
        let room = f.room;
        let snap = f.ilog.observe(snapshot_of(&bv, &d_bv, room));
        if legal.len() == 1 {
            return forced(&legal[0]);
        }
        let cc = merged_classes(cs, &legal, &bv, &d_bv)?;
        if cc.members.len() == 1 {
            return forced(&legal[cc.members[0][0]]);
        }
        check_caps(legal.len(), cc.members.len(), cc.dtype)?;
        let (seq, fight_rows) = {
            let f = self.fight.as_ref().ok_or("no fight")?;
            (f.seq, f.rows)
        };
        if fight_rows >= ct::MAX_FIGHT_DECISIONS {
            return Ok(Next::Terminal(self.cap(rm, "fight_decisions")));
        }
        if self.rows >= ct::MAX_RUN_DECISIONS {
            return Ok(Next::Terminal(self.cap(rm, "run_decisions")));
        }
        let sel = if matches!(cc.dtype, dt::COMBAT_CHOICE | dt::CARD_GRANT) {
            ct::combat_selection(sts2_wire::emit::pending_choice(&bv).as_ref(), &cc.kinds)
        } else {
            Selection::default()
        };
        let d = Decision {
            row: self.rows,
            kind: DecisionKind::Combat,
            dtype: cc.dtype,
            kinds: cc.kinds,
            actions: legal,
            obs_actions: cc.obs_actions,
            classes: cc.classes,
            members: cc.members,
            keys: cc.keys,
            sel,
            event_name: None,
            act: rm.run_state.act_index,
            floor: rm.run_state.total_floor,
            combat: Some(CombatView {
                encounter_id: emit_rm::combat_encounter_id(rm),
                bv,
                d_bv,
                snap,
                room,
                fight: seq,
                turn: rnd,
                fight_row: fight_rows,
            }),
            ooc: None,
        };
        if let Some(f) = self.fight.as_mut() {
            f.rows += 1;
        }
        self.rows += 1;
        Ok(Next::Decision(Box::new(d)))
    }

    /// Start a fight if none is tracked, and roll it when the run moved to
    /// another map node (`_start_fight` / the `where` check).
    fn track_fight(&mut self, rm: &RunManager) {
        let at = (rm.run_state.act_index, emit_rm::map_coord(rm));
        if self.fight.as_ref().is_some_and(|f| f.at != at) {
            self.fight = None;
        }
        if self.fight.is_none() {
            self.start_fight(rm);
        }
    }

    /// The round-reset fallback: a round below the fight's starts a new one.
    fn roll_fight(&mut self, rm: &RunManager, rnd: i64) -> &mut Fight {
        if self.fight.as_ref().is_none_or(|f| rnd < f.round) {
            self.start_fight(rm);
        }
        self.fight.as_mut().expect("a fight was just started")
    }

    fn start_fight(&mut self, rm: &RunManager) {
        self.fight = Some(Fight {
            seq: self.fights,
            room: fight_room(rm),
            at: (rm.run_state.act_index, emit_rm::map_coord(rm)),
            ilog: IntentLog::new(),
            round: 0,
            rows: 0,
        });
        self.fights += 1;
    }

    // ── the end ─────────────────────────────────────────────────────────

    fn end(
        &mut self,
        rm: &RunManager,
        outcome: &'static str,
        won: bool,
        cap: Option<&'static str>,
        error: Option<String>,
    ) -> Terminal {
        self.fight = None;
        let t = Terminal {
            outcome,
            won,
            cap,
            error,
            rows: self.rows,
            steps: self.steps,
            floor: rm.run_state.total_floor,
            act: rm.run_state.act_index,
        };
        self.terminal = Some(t.clone());
        t
    }

    /// End the run as an `error` (records the terminal).
    pub fn error(&mut self, rm: &RunManager, e: String) -> Terminal {
        self.end(rm, "error", false, None, Some(e))
    }

    fn cap(&mut self, rm: &RunManager, reason: &'static str) -> Terminal {
        self.end(rm, "capped", false, Some(reason), None)
    }

    /// `_terminal`: the run stepper's end.
    fn finish(&mut self, rm: &RunManager, outcome: &'static str) -> Next {
        let won = rm.run_state.player_won;
        let outcome = if won { "won" } else { outcome };
        if outcome == "no_actions" {
            let msg = format!(
                "NoActions: no legal action at act {}, floor {}, phase {:?}, step {}",
                rm.run_state.act_index,
                rm.run_state.total_floor,
                phase_str(rm.phase()),
                self.steps
            );
            return Next::Terminal(self.error(rm, msg));
        }
        if outcome == "max_steps" {
            return Next::Terminal(self.cap(rm, "steps"));
        }
        let out = if won { "won" } else { "lost" };
        Next::Terminal(self.end(rm, out, won, None, None))
    }
}

/// What [`E2eEnv::step`] applies for the pending decision.
#[derive(Clone, Debug)]
struct Pending {
    combat: bool,
    actions: Vec<Value>,
    members: Vec<Vec<usize>>,
}

/// The decision driver over one run (`E2EEnv`): a [`Driver`] that owns its
/// run and loops `peek` / `apply` through the forced steps.
#[derive(Clone, Debug)]
pub struct E2eEnv {
    rm: RunManager,
    drv: Driver,
    pending: Option<Pending>,
}

impl E2eEnv {
    /// A fresh campaign run for `seed` ([`new_run`]).
    #[must_use]
    pub fn new(seed: i64) -> Self {
        Self::over_run(new_run(seed), seed)
    }

    /// The env over a run the caller built, at any point of it (see the
    /// module docs for the driver state this cannot recover). `seed` is only
    /// the observation's `seed` field (the tokenizer drops it).
    #[must_use]
    pub fn over_run(rm: RunManager, seed: i64) -> Self {
        Self {
            rm,
            drv: Driver::new(seed),
            pending: None,
        }
    }

    /// The live run.
    #[must_use]
    pub const fn rm(&self) -> &RunManager {
        &self.rm
    }

    /// The driver state.
    #[must_use]
    pub const fn driver(&self) -> &Driver {
        &self.drv
    }

    /// Give the run back.
    #[must_use]
    pub fn into_run_manager(self) -> RunManager {
        self.rm
    }

    /// Stored decisions so far.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.drv.rows()
    }

    /// The terminal, once the run has ended.
    #[must_use]
    pub const fn terminal(&self) -> Option<&Terminal> {
        self.drv.terminal()
    }

    /// Grant picks applied since the open card grant opened.
    pub fn set_grant_picked(&mut self, n: i64) {
        self.drv.set_grant_picked(n);
    }

    /// The live fight's intent log, if a fight is tracked.
    #[must_use]
    pub fn intent_log(&self) -> Option<&IntentLog> {
        self.drv.intent_log()
    }

    /// Replace the live fight's intent log (starting the fight if the run
    /// is in one and none is tracked).
    pub fn set_intent_log(&mut self, ilog: IntentLog) {
        self.drv.set_intent_log(&self.rm, ilog);
    }

    /// [`Driver::observe_combat_position`] on the env's run.
    ///
    /// # Errors
    ///
    /// The combat state could not be emitted.
    pub fn observe_combat_position(&mut self) -> R<()> {
        self.drv.observe_combat_position(&self.rm)
    }

    /// Advance to the first stored decision.
    pub fn reset(&mut self) -> Step {
        self.pending = None;
        self.run_to_decision()
    }

    /// Apply merged class `choice` of the pending decision (its first
    /// member) and advance to the next stored decision.
    ///
    /// # Panics
    ///
    /// Without a pending decision, or with `choice` out of range.
    pub fn step(&mut self, choice: usize) -> Step {
        let p = self
            .pending
            .as_ref()
            .expect("step() without a pending decision");
        let ai = p.members[choice][0];
        self.step_action(ai)
    }

    /// Apply member `ai` of the pending decision's `actions`.
    ///
    /// # Panics
    ///
    /// Without a pending decision, or with `ai` out of range.
    pub fn step_action(&mut self, ai: usize) -> Step {
        let p = self
            .pending
            .take()
            .expect("step_action() without a pending decision");
        assert!(ai < p.actions.len(), "action {ai} out of range");
        let applied = if p.combat {
            combat_to_rm(&p.actions[ai])
                .and_then(|a| self.drv.apply(&mut self.rm, &a, ApplyKind::Combat))
        } else {
            self.drv.apply(
                &mut self.rm,
                &p.actions[ai],
                ApplyKind::Run { strict: true },
            )
        };
        if let Err(e) = applied {
            return Step::Terminal(self.drv.error(&self.rm, e));
        }
        self.run_to_decision()
    }

    /// `_drive`: peek, apply the forced steps, stop at a decision or the end.
    fn run_to_decision(&mut self) -> Step {
        loop {
            match self.drv.peek(&mut self.rm) {
                Next::Forced(f) => {
                    if let Err(e) = self.drv.apply(&mut self.rm, &f.action, f.kind) {
                        return Step::Terminal(self.drv.error(&self.rm, e));
                    }
                }
                Next::Decision(d) => {
                    self.pending = Some(Pending {
                        combat: d.is_combat(),
                        actions: d.actions.clone(),
                        members: d.members.clone(),
                    });
                    return Step::Decision(d);
                }
                Next::Terminal(t) => return Step::Terminal(t),
            }
        }
    }
}

/// Whether the run is in a fight: it offers combat actions and has a
/// combat state.
fn in_combat(rm: &RunManager) -> bool {
    is_combat(&available(rm)) && rm.combat.is_some()
}

/// `check_caps`: 255 members, 64 merged classes (SCHEMA.md §7).
fn check_caps(n_members: usize, n_classes: usize, dtype: u8) -> R<()> {
    if n_members > ct::MAX_MEMBERS || n_classes > ct::MAX_OPTIONS {
        return Err(format!(
            "SchemaError: {}: {n_members} members / {n_classes} classes exceed {} / {}",
            ct::dtype_name(dtype),
            ct::MAX_MEMBERS,
            ct::MAX_OPTIONS
        ));
    }
    Ok(())
}

/// The forced step that resolves Claws' optional "up to 6 into Maul" select
/// (GH #1353) without a decision: the first offered card until the cap, then
/// the confirm, which is always last. That is the first six candidates in deck
/// order, the pick the engine made before the select surfaced. The schema
/// (`ironclad-e2e-v111`) has no kind for the confirm, so the select stays out
/// of the policy until a schema revision adds one (`_claws_auto_pick` in
/// `ppo/e2e/env.py`).
pub(crate) fn claws_auto_pick(rm: &RunManager, actions: &[Value]) -> Option<Value> {
    rm.run_state
        .pending_claws_selection
        .as_ref()
        .and_then(|_| actions.first().cloned())
}

pub(crate) fn available(rm: &RunManager) -> Vec<Value> {
    match emit_rm::available_actions(rm) {
        Value::Array(a) => a,
        _ => Vec::new(),
    }
}

pub(crate) fn round_of(d_bv: &Value) -> i64 {
    get(d_bv, "round")
        .filter(|v| truthy(v))
        .and_then(as_int)
        .unwrap_or(0)
}

pub(crate) fn snapshot_of(bv: &CombatState, d_bv: &Value, room: &str) -> Value {
    let pc = sts2_wire::emit::pending_choice(bv);
    snapshot::snapshot_from_parts(
        d_bv,
        &sts2_wire::emit::card_damage_preview(bv),
        &sts2_wire::emit::card_block_preview(bv),
        pc.as_ref(),
        room,
    )
}

#[cfg(test)]
mod face_var_tests {
    use super::*;

    /// Vigorous adds its amount until a card's first play; an Anger clone
    /// keeps it, so a spent and a fresh copy are two classes (#1097: eval seed
    /// 160000444). Mirrors `test_a_spent_one_shot_enchantment_splits_the_play_class`.
    #[test]
    fn a_spent_one_shot_enchantment_is_a_face_var() {
        let base = serde_json::json!({"id": "ANGER", "kind": "card", "enchantment": "Vigorous", "enchantment_amount": 8});
        let mut fresh = base.clone();
        fresh["combat_vars"] = serde_json::json!({"generated_in_combat": 1});
        let mut spent = base;
        spent["combat_vars"] = serde_json::json!({"enchantment_used": 1});
        assert!(face_vars(&fresh).is_empty());
        assert_eq!(
            Value::Object(face_vars(&spent)),
            serde_json::json!({"enchantment_used": 1})
        );
    }
}

#[cfg(test)]
mod claws_tests {
    use super::*;
    use sts2_core::cards::CardId;
    use sts2_core::relics::RelicId;

    /// GH #1353: the schema has no kind for Claws' confirm, so the env
    /// resolves the select as forced steps: the first six cards, then the
    /// confirm. Mirrors `test_the_e2e_env_takes_the_first_six_without_a_decision`.
    #[test]
    fn claws_is_six_forced_picks_and_a_forced_confirm() {
        let mut rm = new_run(7);
        let deck: Vec<&str> = ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"].repeat(4);
        sts2_wire::graft::set_master_deck(&mut rm, &json!(deck)).expect("deck");
        rm.run_state.obtain_relic(RelicId::Claws);
        assert!(rm.run_state.pending_claws_selection.is_some());
        let mut driver = Driver::new(7);
        let mut forced = Vec::new();
        while rm.run_state.pending_claws_selection.is_some() {
            let Next::Forced(f) = driver.peek(&mut rm) else {
                panic!("Claws' select must be a forced step");
            };
            forced.push(action_name(&f.action).to_owned());
            driver.apply(&mut rm, &f.action, f.kind).expect("apply");
        }
        let mut expect = vec!["transform_deck_card".to_owned(); 6];
        expect.push("confirm_deck_selection".to_owned());
        assert_eq!(forced, expect);
        let ids: Vec<CardId> = rm.run_state.master_deck.iter().map(|c| c.card_id).collect();
        assert_eq!(&ids[..2], &[CardId::StrikeIronclad, CardId::DefendIronclad]);
        assert!(ids[2..].iter().all(|&c| c == CardId::Maul));
    }
}
