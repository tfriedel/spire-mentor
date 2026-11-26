//! Instinct's out-of-fight screens (GH #1426 step 3; were
//! `realgame_instinct_screens.py`): each rebuilds the run behind the screen
//! (`graft::graft_run`), opens the screen in the engine, asks the student
//! (an [`Evaluator`]) and maps its classes back onto the screen's buttons
//! (`instinct::screen_advice`).

use std::collections::BTreeSet;

use serde_json::{json, Value};
use sts2_core::run_manager::RunManager;

use crate::display_names::Names;
use crate::graft::{graft_run, GraftError};
use crate::i18n::{map_label, tr, trf};
use crate::instinct::{screen_advice, Pick};
use crate::py;

/// The student at one rebuilt position: `InstinctAdvisor.evaluate`.
pub trait Evaluator {
    /// Peek `rm` with this run's driver state and run the student;
    /// `expect` is the decision types this screen can be.
    ///
    /// # Errors
    ///
    /// The player-facing reason Instinct cannot answer here.
    fn evaluate(
        &mut self,
        rm: &mut RunManager,
        expect: &[&str],
    ) -> std::result::Result<Pick, String>;

    /// The picks the open card grant already took (`R_SEL_PICKED`): a
    /// rebuild that replays them in the engine does so behind the
    /// driver's back, which counts only the picks it sees taken (GH #1677).
    fn set_grant_picked(&mut self, _n: i64) {}
}

type Result<T> = std::result::Result<T, GraftError>;

fn unavailable(msg: impl Into<String>) -> GraftError {
    GraftError::Unavailable(msg.into())
}

/// `norm_id`: an id in any spelling (`STRIKE_IRONCLAD`, `StrikeIronclad`,
/// `CARD.STRIKE_IRONCLAD`, `RELIC.ANCHOR`) as one comparable key.
#[must_use]
pub fn norm_id(raw: &Value) -> String {
    let s = if py::truthy(raw) {
        py::str_of(raw)
    } else {
        String::new()
    };
    let tail = s.rsplit('.').next().unwrap_or("");
    tail.to_uppercase()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .filter(|c| !c.is_ascii_lowercase())
        .collect()
}

fn norm_str(s: &str) -> String {
    norm_id(&Value::String(s.to_string()))
}

/// `registry_id`: a wire card id in the engine registry's spelling
/// (`SETUP_STRIKE` -> `SETUP_STRIKE_CARD`), else the id as given.
#[must_use]
pub fn registry_id(raw: &Value) -> String {
    sts2_workbench::combat_import::registry_card_id(&py::str_of(raw), "$.card").unwrap_or_else(
        |_| {
            if py::truthy(raw) {
                py::str_of(raw)
            } else {
                String::new()
            }
        },
    )
}

/// `_label`: a card or option's display text.
fn label(c: &Value) -> String {
    ["name", "label", "id"]
        .iter()
        .map(|k| py::get(c, k))
        .find(|v| py::truthy(v))
        .map_or_else(|| "?".to_string(), py::str_of)
}

/// `str(a or b)` over two values (`"None"` when neither is truthy).
fn str_or(a: &Value, b: &Value) -> String {
    py::str_of(if py::truthy(a) { a } else { b })
}

/// `int(d.get("index", default))`.
fn index_or(d: &Value, default: usize) -> i64 {
    d.get("index").map_or(default as i64, py::int_of)
}

/// The run-action dict of class `ci`'s representative member.
fn first_action(pick: &Pick, ci: usize) -> Value {
    let actions = py::get(
        &py::list(&pick.decision, "classes")
            .get(ci)
            .cloned()
            .unwrap_or(Value::Null),
        "actions",
    )
    .clone();
    py::as_list(&actions)
        .first()
        .filter(|a| py::truthy(a))
        .cloned()
        .unwrap_or_else(|| json!({}))
}

/// The dicts in `state[key]`.
fn dicts<'a>(state: &'a Value, key: &str) -> Vec<&'a Value> {
    py::list(state, key)
        .iter()
        .filter(|v| v.is_object())
        .collect()
}

fn choose(index: i64) -> Value {
    json!({"action": "choose", "index": index})
}

fn advise(pick: &Pick, rows: &[Option<(Value, String)>]) -> Result<Value> {
    screen_advice(pick, rows, "").map_err(GraftError::Unavailable)
}

/// A card reward (or an out-of-fight choose-a-card screen).
///
/// # Errors
///
/// Instinct cannot answer here ([`GraftError::Unavailable`]), or the screen
/// is malformed ([`GraftError::Value`]).
pub fn card_reward(
    ev: &mut dyn Evaluator,
    state: &Value,
    version: &str,
    last_room: Option<&str>,
) -> Result<Value> {
    let surface = py::get(state, "surface").as_str();
    if surface == Some("choose_a_card") && !py::get(state, "combat").is_null() {
        return Err(unavailable(tr(
            "a card offered into the fight needs the fight mirror",
        )));
    }
    let cards = dicts(state, "cards");
    if cards.is_empty() {
        return Err(unavailable(tr("no cards on this reward")));
    }
    let (mut rm, _notes) = graft_run(state, version)?;
    // The room the game names first: Neow's card (an event room, GH #1664)
    // has no fight before it. A later act's entrance stands in no room in
    // the engine, whatever the game names (GH #1646, #1732): Glass Eye's
    // card rewards at Orobas read X_ROOM none natively (Spire Bench seed
    // 180001539), as `graft_screen_run` keeps a deck screen there.
    let at_later_entrance = rm.run_state.act_floor == 0 && rm.run_state.current_room.is_none();
    if !at_later_entrance {
        let room = crate::graft::live_room(state)
            .or(last_room)
            .unwrap_or("monster");
        sts2_wire::graft::set_room_type(&mut rm, room).map_err(GraftError::Value)?;
    }
    // Out of a fight, a choose-a-card screen adds to the deck (Lead
    // Paperweight, Hefty Tablet): the engine's run-effect choose screen,
    // offered as the CARD_REWARD the student saw in training.
    let specs: Vec<Value> = cards.iter().map(|c| crate::graft::card_spec(c)).collect();
    let can_skip = state.get("can_skip").is_none_or(py::truthy);
    sts2_wire::graft::open_card_offer(
        &mut rm,
        &Value::Array(specs),
        can_skip,
        surface == Some("choose_a_card"),
        None,
    )
    .map_err(GraftError::Value)?;
    let offered: Vec<String> = sts2_wire::emit::run_manager::card_reward_option_ids(&rm)
        .iter()
        .map(|x| norm_str(x))
        .collect();
    let want: Vec<String> = cards
        .iter()
        .map(|c| norm_str(&registry_id(py::get(c, "id"))))
        .collect();
    if offered != want {
        return Err(unavailable(tr(
            "the rebuilt reward does not match the screen",
        )));
    }
    let pick = ev
        .evaluate(&mut rm, &["CARD_REWARD"])
        .map_err(GraftError::Unavailable)?;
    let rows: Vec<Option<(Value, String)>> = (0..pick.probs.len())
        .map(|ci| {
            let a = first_action(&pick, ci);
            // Pael's Wing's sacrifice: not a button on this screen.
            if py::get(&a, "action").as_str() != Some("pick_card_reward") {
                return None;
            }
            let idx = py::get(&a, "idx");
            if idx.is_null() {
                return Some((json!({"action": "skip"}), tr("skip").to_string()));
            }
            let i = py::int_of(idx);
            let c = usize::try_from(i).ok().and_then(|i| cards.get(i))?;
            Some((choose(index_or(c, i as usize)), label(c)))
        })
        .collect();
    advise(&pick, &rows)
}

/// A Scroll Boxes pack choice (GH #978).
///
/// # Errors
///
/// As [`card_reward`].
pub fn card_bundle(ev: &mut dyn Evaluator, state: &Value, version: &str) -> Result<Value> {
    let bundles = dicts(state, "bundles");
    if bundles.is_empty() {
        return Err(unavailable(tr("no packs on this screen")));
    }
    let (mut rm, _notes) = graft_run(state, version)?;
    let packs: Vec<Value> = bundles
        .iter()
        .map(|b| {
            Value::Array(
                py::list(b, "cards")
                    .iter()
                    .map(crate::graft::card_spec)
                    .collect(),
            )
        })
        .collect();
    sts2_wire::graft::open_card_offer(
        &mut rm,
        &json!([]),
        false,
        false,
        Some(&Value::Array(packs)),
    )
    .map_err(GraftError::Value)?;
    let pick = ev
        .evaluate(&mut rm, &["CARD_BUNDLE"])
        .map_err(GraftError::Unavailable)?;
    let rows: Vec<Option<(Value, String)>> = (0..pick.probs.len())
        .map(|ci| {
            let i = first_action(&pick, ci).get("idx").and_then(Value::as_i64)?;
            let b = usize::try_from(i).ok().and_then(|i| bundles.get(i))?;
            let names: Vec<String> = py::list(b, "cards").iter().map(label).collect();
            let text = if names.is_empty() {
                format!("pack {i}")
            } else {
                names.join(" + ")
            };
            Some((choose(index_or(b, i as usize)), text))
        })
        .collect();
    advise(&pick, &rows)
}

/// The rest site.
///
/// # Errors
///
/// As [`card_reward`].
pub fn rest_site(ev: &mut dyn Evaluator, state: &Value, version: &str) -> Result<Value> {
    let options = dicts(state, "options");
    if options.is_empty() {
        return Err(unavailable(tr("no rest-site options")));
    }
    let (mut rm, _notes) = graft_run(state, version)?;
    let offer = |rm: &mut RunManager, taken: &[String]| -> Result<Vec<String>> {
        Ok(sts2_wire::graft::enter_rest_site(rm, taken)
            .map_err(GraftError::Value)?
            .iter()
            .map(|x| norm_str(x))
            .collect())
    };
    let mut ids = offer(&mut rm, &[])?;
    // The game takes a used option off the screen and keeps the others
    // (Miniature Tent, `RestSiteSynchronizer.ChooseOption`). An option the
    // engine's fresh site offers and the screen no longer shows was taken
    // earlier this visit; the engine then offers leaving in its place.
    let shown: Vec<String> = options.iter().map(|o| norm_id(py::get(o, "id"))).collect();
    let taken: Vec<String> = ids
        .iter()
        .filter(|id| !shown.contains(id))
        .cloned()
        .collect();
    if !taken.is_empty() && taken.len() < ids.len() {
        ids = offer(&mut rm, &taken)?;
    }
    let enabled = |o: &&&Value| o.get("enabled").is_none_or(py::truthy);
    let by_id = |id: &str| -> Option<&Value> {
        options
            .iter()
            .filter(enabled)
            .rfind(|o| norm_id(py::get(o, "id")) == id)
            .copied()
    };
    let pick = ev
        .evaluate(&mut rm, &["REST"])
        .map_err(GraftError::Unavailable)?;
    // Leaving with options still open (Miniature Tent, the engine's
    // `leave_shop` here): the screen's own proceed (`can_leave`). Dropped,
    // the student's pick fell to the next option, a rest it did not want
    // (Spire Bench seed 180000226, GH #1833).
    let can_leave = py::truthy(py::get(state, "can_leave"));
    let rows: Vec<Option<(Value, String)>> = (0..pick.probs.len())
        .map(|ci| {
            let a = first_action(&pick, ci);
            if py::get(&a, "action").as_str() == Some("leave_shop") {
                return can_leave.then(|| (json!({"action": "leave"}), "leave".to_string()));
            }
            let i = a.get("idx").and_then(Value::as_i64)?;
            let id = usize::try_from(i).ok().and_then(|i| ids.get(i))?;
            let o = by_id(id)?;
            Some((
                choose(py::int_of(py::get(o, "index"))),
                str_or(py::get(o, "label"), py::get(o, "id")),
            ))
        })
        .collect();
    advise(&pick, &rows)
}

/// A relic offer answered as the engine's boss-relic pick: the student's
/// classes mapped onto `offered`, each labelled by `label`.
fn relic_rows(
    ev: &mut dyn Evaluator,
    rm: &mut RunManager,
    offered: &[&Value],
    label: impl Fn(&Value) -> String,
) -> Result<Value> {
    let pick = ev
        .evaluate(rm, &["BOSS_RELIC"])
        .map_err(GraftError::Unavailable)?;
    let rows: Vec<Option<(Value, String)>> = (0..pick.probs.len())
        .map(|ci| {
            let i = first_action(&pick, ci).get("idx").and_then(Value::as_i64)?;
            let o = usize::try_from(i).ok().and_then(|i| offered.get(i))?;
            Some((choose(index_or(o, i as usize)), label(o)))
        })
        .collect();
    advise(&pick, &rows)
}

fn open_relics(rm: &mut RunManager, ids: &[String]) -> Result<()> {
    sts2_wire::graft::open_boss_relic(rm, ids).map_err(|e| {
        unavailable(trf(
            "the engine cannot offer these relics ({e})",
            &[("e", &e)],
        ))
    })
}

/// The boss relic screen.
///
/// # Errors
///
/// As [`card_reward`].
pub fn boss_relic(ev: &mut dyn Evaluator, state: &Value, version: &str) -> Result<Value> {
    let offered: Vec<&Value> = dicts(state, "options")
        .into_iter()
        .filter(|o| o.get("enabled").is_none_or(py::truthy))
        .collect();
    let ids: Vec<String> = offered
        .iter()
        .map(|o| {
            let id = py::get(py::get(o, "relic"), "id");
            if py::truthy(id) {
                py::str_of(id)
            } else {
                String::new()
            }
        })
        .collect();
    if offered.is_empty() || ids.iter().any(String::is_empty) {
        return Err(unavailable(tr("no relics on this screen")));
    }
    let (mut rm, _notes) = graft_run(state, version)?;
    open_relics(&mut rm, &ids)?;
    relic_rows(ev, &mut rm, &offered, |o| {
        str_or(py::get(py::get(o, "relic"), "name"), py::get(o, "label"))
    })
}

/// An Act 2 / 3 Ancient's relic offer: the engine's boss-relic pick.
///
/// # Errors
///
/// As [`card_reward`].
pub fn ancient_relic_choice(
    ev: &mut dyn Evaluator,
    state: &Value,
    version: &str,
    options: &[&Value],
) -> Result<Value> {
    let offered: Vec<&Value> = options
        .iter()
        .filter(|o| o.get("enabled").is_none_or(py::truthy) && py::truthy(py::get(o, "relic_id")))
        .copied()
        .collect();
    let (mut rm, _notes) = graft_run(state, version)?;
    let ids: Vec<String> = offered
        .iter()
        .map(|o| py::str_of(py::get(o, "relic_id")))
        .collect();
    open_relics(&mut rm, &ids)?;
    relic_rows(ev, &mut rm, &offered, |o| {
        str_or(py::get(o, "label"), py::get(o, "relic_id"))
    })
}

/// `sts2_realgame.potion_swap_group`: the first full-belt potion reward's
/// `potion_swap` options, decline first.
#[must_use]
pub fn potion_swap_group(options: &[Value]) -> Vec<&Value> {
    let button = |o: &Value| {
        let b = py::get(o, "reward_button");
        if py::truthy(b) {
            py::int_of(b)
        } else {
            0
        }
    };
    let swaps: Vec<&Value> = options
        .iter()
        .filter(|o| py::get(o, "action").as_str() == Some("potion_swap"))
        .collect();
    let Some(first) = swaps.iter().map(|o| button(o)).min() else {
        return Vec::new();
    };
    let mut group: Vec<&Value> = swaps.into_iter().filter(|o| button(o) == first).collect();
    group.sort_by_key(|o| match py::get(o, "discard_slot") {
        Value::Null => -1,
        v => py::int_of(v),
    });
    group
}

/// The full-belt potion swap.
///
/// # Errors
///
/// As [`card_reward`].
pub fn potion_swap(
    ev: &mut dyn Evaluator,
    state: &Value,
    version: &str,
    names: &Names,
) -> Result<Value> {
    let group = potion_swap_group(py::list(state, "options"));
    if group.is_empty() {
        return Err(unavailable(tr("no potion swap on this screen")));
    }
    let id = py::get(py::get(group[0], "reward"), "id");
    let potion = if py::truthy(id) {
        py::str_of(id)
    } else {
        String::new()
    };
    if potion.is_empty() {
        return Err(unavailable(tr("the screen does not name the potion")));
    }
    let (mut rm, _notes) = graft_run(state, version)?;
    sts2_wire::graft::queue_potion_swap(&mut rm, &potion).map_err(GraftError::Value)?;
    let pick = ev
        .evaluate(&mut rm, &["POTION_SWAP"])
        .map_err(GraftError::Unavailable)?;
    // `{discard_slot: option}`: the last option with a slot wins.
    let by_slot = |slot: &Value| {
        group
            .iter()
            .rev()
            .find(|o| py::eq(py::get(o, "discard_slot"), slot))
            .copied()
    };
    let rs_potions = py::list(py::get(state, "run_state"), "potions");
    let belt: &[Value] = if rs_potions.is_empty() {
        py::list(state, "potions")
    } else {
        rs_potions
    };
    let rows: Vec<Option<(Value, String)>> = (0..pick.probs.len())
        .map(|ci| {
            let a = first_action(&pick, ci);
            let o = by_slot(a.get("discard_slot")?)?;
            Some((
                choose(py::int_of(py::get(o, "index"))),
                names.potion_swap_label(o, belt),
            ))
        })
        .collect();
    let mut advice = advise(&pick, &rows)?;
    // `best.label` stays the mod's own spelling (`discard <POTION_ID>`): the
    // takeover cross-checks it (SimAdvisor.ReconcileIndex).
    let best_index = advice["best"].get("index").cloned().unwrap_or(Value::Null);
    if let Some(chosen) = group
        .iter()
        .find(|o| py::eq(py::get(o, "index"), &best_index))
    {
        let l = py::get(chosen, "label");
        if py::truthy(l) {
            advice["best"]["label"] = json!(py::str_of(l));
        }
    }
    Ok(advice)
}

/// A card's identity on a deck screen: `(norm id, upgraded, norm
/// enchantment or "")` (`_card_key`).
pub type CardKey = (String, bool, String);

fn card_key(card_id: &Value, upgraded: bool, enchantment: &Value) -> CardKey {
    let ench = if py::truthy(enchantment) {
        norm_id(enchantment)
    } else {
        String::new()
    };
    (norm_id(card_id), upgraded, ench)
}

/// `wire_card_key`: a wire card's identity, in the engine registry's
/// spelling.
fn wire_card_key(card: &Value) -> CardKey {
    let upgraded = match py::get(card, "upgrade_level") {
        Value::Null => py::truthy(py::get(card, "upgraded")),
        level => py::truthy(level),
    };
    card_key(
        &json!(registry_id(py::get(card, "id"))),
        upgraded,
        py::get(card, "enchantment"),
    )
}

/// `deck_keys`: the engine's master deck as identity keys, deck order.
#[must_use]
pub fn deck_keys(rm: &RunManager) -> Vec<CardKey> {
    let detailed = sts2_wire::emit::run_manager::master_deck_detailed(rm);
    let enchants = sts2_wire::emit::run_manager::master_deck_enchantments(rm);
    detailed
        .iter()
        .enumerate()
        .map(|(i, (name, upgraded))| {
            let ench = enchants
                .get(i)
                .cloned()
                .flatten()
                .map_or(Value::Null, |(id, _)| json!(id));
            card_key(&json!(name), *upgraded, &ench)
        })
        .collect()
}

/// `picked_deck_card`: the deck card `action` takes away (removes /
/// upgrades / transforms), found by applying it to a copy of the run: the
/// card in the deck before and not after is the pick, whatever index space
/// the action uses.
fn picked_deck_card(rm: &RunManager, action: &Value) -> Option<CardKey> {
    let mut before: Vec<CardKey> = deck_keys(rm);
    let mut probe = rm.clone();
    let act = sts2_wire::emit::run_action_from_value(action).ok()?;
    let _ = probe.take_action(act);
    // `Counter(before) - Counter(after)`: what is left of `before` once
    // every card still in the deck is struck off.
    for k in deck_keys(&probe) {
        if let Some(i) = before.iter().position(|b| *b == k) {
            before.remove(i);
        }
    }
    let mut gone: Vec<CardKey> = Vec::new();
    for k in before {
        if !gone.contains(&k) {
            gone.push(k);
        }
    }
    if gone.len() == 1 {
        gone.pop()
    } else {
        None
    }
}

/// `_find_card`: the first wire card with identity `key`, else (the
/// engine and the wire may disagree on an enchantment) the first with its
/// id and upgrade.
fn find_card(cards: &[&Value], key: &CardKey) -> Option<usize> {
    let keys: Vec<CardKey> = cards.iter().map(|c| wire_card_key(c)).collect();
    keys.iter()
        .position(|k| k == key)
        .or_else(|| keys.iter().position(|k| k.0 == key.0 && k.1 == key.1))
}

/// The screen's cards, each with its deck card's enchantment. A deck screen
/// lists the deck's eligible cards in deck order (C# `FromDeckForUpgrade`
/// and the others filter `PileType.Deck`), but the screen's cards carry no
/// enchantment, so two copies apart only by one (an Adroit Pommel Strike and
/// a plain one) read the same and a pick of the plain one upgraded the
/// enchanted one (GH #1763). The k-th copy of a card on the screen is the
/// k-th copy in the master deck: a screen can put other cards first (a
/// removal lists the curses first) or leave some out (GH #1767). Left as
/// they are when the deck holds fewer copies than the screen.
///
/// An enchant screen offers no enchanted card at all: `EnchantmentModel.
/// CanEnchant` refuses a card whose `Enchantment` is set unless it is the
/// same stackable kind, and no v0.111.0 enchantment is stackable. Its k-th
/// Defend is the k-th UNenchanted one, so pairing it with the deck's k-th
/// Defend stamped a Spiral (Nimble, Adroit) the card does not have onto the
/// first one: the engine's pick, the first plain Defend, then matched the
/// second, and the game enchanted another copy than the engine (corpus seeds
/// 180000183, 180000190, 180001030, 180001720: the enchanted Defend drew a
/// turn early or late).
fn with_deck_enchantments(state: &Value) -> Vec<Value> {
    let cards: Vec<Value> = dicts(state, "cards").into_iter().cloned().collect();
    if cards.iter().any(|c| c.get("enchantment").is_some()) {
        return cards;
    }
    let prompt = py::get(state, "prompt_key");
    if py::truthy(prompt) && py::str_of(prompt).to_uppercase() == "TO_ENCHANT" {
        return cards;
    }
    let deck = dicts(py::get(state, "run_state"), "master_deck");
    let key = |c: &Value| {
        let k = wire_card_key(c);
        (k.0, k.1)
    };
    let mut seen: std::collections::BTreeMap<(String, bool), usize> =
        std::collections::BTreeMap::new();
    let mut out = Vec::with_capacity(cards.len());
    for c in &cards {
        let k = key(c);
        let nth = seen.entry(k.clone()).or_insert(0);
        let Some(d) = deck.iter().filter(|d| key(d) == k).nth(*nth) else {
            return cards;
        };
        *nth += 1;
        let mut c = c.clone();
        if let Some(e) = d.get("enchantment").filter(|e| py::truthy(e)) {
            c["enchantment"] = e.clone();
        }
        out.push(c);
    }
    out
}

/// `int(d.get(a) or d.get(b) or default)`.
fn int_or2(d: &Value, a: &str, b: &str, default: i64) -> i64 {
    [a, b]
        .iter()
        .map(|k| py::get(d, k))
        .find(|v| py::truthy(v))
        .map_or(default, py::int_of)
}

/// prompt_key -> what the screen does (`realgame_ppo._PROMPT_SURFACES`' twin).
fn prompt_kind(prompt: &str) -> Option<&'static str> {
    Some(match prompt {
        "TO_REMOVE" => "remove",
        "TO_TRANSFORM" | "CLAWS.SELECTIONSCREENPROMPT" => "transform",
        "TO_ENCHANT" => "enchant",
        "TO_UPGRADE" => "upgrade",
        "TO_OBTAIN"
        | "BRAIN_LEECH.PAGES.SHARE_KNOWLEDGE.SELECTIONSCREENPROMPT"
        | "ROOM_FULL_OF_CHEESE.PAGES.GORGE.SELECTIONSCREENPROMPT"
        | "SEA_GLASS.SELECTIONSCREENPROMPT"
        // Dolly's Mirror copies a deck card: the engine's card grant over
        // the non-Quest deck (`listener_dollys_mirror`, GH #1750).
        | "DOLLYS_MIRROR.SELECTIONSCREENPROMPT" => "grant",
        _ => return None,
    })
}

/// The enchantment an enchant screen applies (`NIMBLE`), or `None`. The mod's
/// `pending_enchantment` when the key is present, final even as `null` (a
/// `source_model` beside it may be an earlier selection's; GH #574); from a mod
/// before the field, `source_model` when `source_kind` is absent or
/// `"enchantment"`. The PPO lane reads it the same way
/// (`realgame_ppo._wire_pending_enchantment`).
fn screen_enchantment(state: &Value) -> Option<String> {
    if let Some(pending) = state.get("pending_enchantment") {
        return py::truthy(pending).then(|| py::str_of(pending));
    }
    let model = py::get(state, "source_model");
    let kind = py::get(state, "source_kind");
    let kind = if py::truthy(kind) {
        py::str_of(kind)
    } else {
        "enchantment".into()
    };
    (py::truthy(model) && kind.to_lowercase() == "enchantment").then(|| py::str_of(model))
}

/// The card types the offered cards have, when they restrict a pick: one
/// (Self-Help Book, Momentum, Swift) or two (Royal Stamp's Royally Approved
/// takes an Attack or a Skill, `RoyallyApproved.CanEnchantCardType`). All
/// three playable types, or any other, restrict nothing.
fn offered_card_types(cards: &[&Value]) -> (Option<String>, Option<String>) {
    let mut types: Vec<String> = Vec::new();
    for c in cards {
        let t = py::get(c, "type");
        let t = if py::truthy(t) {
            py::str_of(t).to_lowercase()
        } else {
            String::new()
        };
        if !types.contains(&t) {
            types.push(t);
        }
    }
    let playable = |t: &String| matches!(t.as_str(), "attack" | "skill" | "power");
    match types.as_slice() {
        [t] if playable(t) => (Some(t.clone()), None),
        [t, u] if playable(t) && playable(u) => (Some(t.clone()), Some(u.clone())),
        _ => (None, None),
    }
}

/// Claws' "up to 6 into Maul" screen, played as Instinct's env plays it:
/// the first offered card until six are taken, then confirm (GH #1353).
fn claws_rule(state: &Value, cards: &[&Value]) -> Value {
    let taken = int_or2(state, "selected_count", "selected_count", 0).max(0);
    let room = (int_or2(state, "max_select", "max_count", 0) - taken).max(0);
    let options: Vec<Value> = cards
        .iter()
        .enumerate()
        .map(|(j, c)| json!({"label": label(c), "is_best": room > 0 && j == 0, "index": index_or(c, j), "action": "choose"}))
        .collect();
    let best = match cards.first() {
        Some(c) if room > 0 => {
            json!({"action": "choose", "index": index_or(c, 0), "label": label(c)})
        }
        _ => json!({"action": "skip", "label": tr("confirm")}),
    };
    json!({
        "note": tr("Instinct (e2e net, no search) · Claws by rule: the first six cards, as its env plays them"),
        "options": options,
        "best": best,
    })
}

/// Python's `cards[i]`: a negative index counts from the end.
fn py_index<'a>(cards: &[&'a Value], i: i64) -> Option<&'a Value> {
    let n = cards.len() as i64;
    let j = if i < 0 { n + i } else { i };
    usize::try_from(j).ok().and_then(|j| cards.get(j)).copied()
}

/// The run standing on the smith's pick: rest site, Smith taken.
fn smith_rm(state: &Value, version: &str) -> Result<RunManager> {
    let (mut rm, _notes) = graft_run(state, version)?;
    let ids: Vec<String> = sts2_wire::graft::enter_rest_site(&mut rm, &[])
        .map_err(GraftError::Value)?
        .iter()
        .map(|x| norm_str(x))
        .collect();
    let Some(i) = ids.iter().position(|x| x == "SMITH") else {
        return Err(unavailable(tr("the rebuilt rest site has no Smith")));
    };
    let act = sts2_wire::emit::run_action_from_value(&json!({"action": "rest_option", "idx": i}))
        .map_err(|e| GraftError::Value(e.to_string()))?;
    let _ = rm.take_action(act);
    Ok(rm)
}

/// The engine room (`set_room_type` spelling) a deck-card screen stands in.
/// The graft has no current room, and the student reads it (RUNCTX
/// `X_ROOM`): an event's upgrade or grant read as an "unknown" room before
/// (GH #1197). The mod names the live room (`current_room`, the game's
/// `RoomType`); a mod without the field falls back to the screen Instinct
/// last answered, which misses screens it never saw (a relic's grant after
/// a card reward).
fn screen_room(state: &Value, last_screen: Option<&str>) -> Option<&'static str> {
    if state
        .get("current_room")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
    {
        return crate::graft::live_room(state);
    }
    match last_screen? {
        "event" => Some("event"),
        "shop" => Some("shop"),
        "rest_site" => Some("rest_site"),
        "treasure" => Some("treasure"),
        _ => None,
    }
}

/// [`graft_run`] standing in the deck-card screen's room ([`screen_room`]).
fn graft_screen_run(state: &Value, version: &str, last_screen: Option<&str>) -> Result<RunManager> {
    let (mut rm, _notes) = graft_run(state, version)?;
    // A later act's entrance stands in no room in the engine (GH #1646, #1732):
    // an Ancient's relic that opens a deck screen there (Biiig Hug's
    // removal, Electric Shrymp's enchant) leaves it in none, whatever the
    // screen before it was.
    let at_later_entrance = rm.run_state.act_floor == 0 && rm.run_state.current_room.is_none();
    if !at_later_entrance {
        if let Some(room) = screen_room(state, last_screen) {
            sts2_wire::graft::set_room_type(&mut rm, room).map_err(GraftError::Value)?;
        }
    }
    Ok(rm)
}

/// Whether `state` names `event` (`event_id`, any case or separators).
fn is_event(state: &Value, event: &str) -> bool {
    let id = py::str_of(py::get(state, "event_id"));
    let norm: String = id
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_uppercase();
    norm == event
}

/// An out-of-fight card select: remove, transform, enchant, upgrade (a
/// smith after a rest site), or a card grant.
///
/// `last_screen` is the previous request's `advice_for`; `shop_removal` the
/// deck card Instinct removed in the shop (the shop chose it, this screen
/// only executes it).
///
/// # Errors
///
/// As [`card_reward`].
#[allow(clippy::too_many_lines)]
pub fn deck_select(
    ev: &mut dyn Evaluator,
    state: &Value,
    version: &str,
    last_screen: Option<&str>,
    shop_removal: Option<&CardKey>,
) -> Result<Value> {
    let enriched = with_deck_enchantments(state);
    let all_cards: Vec<&Value> = enriched.iter().collect();
    // A pick-N screen asked one pick at a time (`picked_indices`: the
    // earlier picks, positions in `cards`, in order).
    let earlier: Vec<usize> = py::list(state, "picked_indices")
        .iter()
        .filter_map(|v| v.as_u64().and_then(|u| usize::try_from(u).ok()))
        .collect();
    let cards: Vec<&Value> = all_cards
        .iter()
        .enumerate()
        .filter(|(k, _)| !earlier.contains(k))
        .map(|(_, c)| *c)
        .collect();
    let prompt = py::get(state, "prompt_key");
    let prompt = if py::truthy(prompt) {
        py::str_of(prompt).to_uppercase()
    } else {
        String::new()
    };
    if prompt == "CLAWS.SELECTIONSCREENPROMPT" {
        return Ok(claws_rule(state, &cards));
    }
    if cards.is_empty() {
        return Err(unavailable(tr("no cards on this screen")));
    }
    let Some(kind) = prompt_kind(&prompt) else {
        return Err(unavailable(tr(
            "Instinct does not know what this card screen does",
        )));
    };
    let n = int_or2(state, "max_select", "max_count", 1).max(1);
    let lo = int_or2(state, "min_select", "min_count", 0);
    let multi = n > 1 && state.get("picked_indices").is_some();

    if kind == "remove" && last_screen == Some("shop") {
        if let Some(i) = shop_removal.and_then(|k| find_card(&cards, k)) {
            let c = cards[i];
            let options: Vec<Value> = cards
                .iter()
                .enumerate()
                .map(|(j, x)| json!({"label": label(x), "is_best": j == i, "index": index_or(x, j), "action": "choose"}))
                .collect();
            return Ok(json!({
                "note": tr("Instinct (e2e net, no search) · the card it picked in the shop"),
                "options": options,
                "best": {"action": "choose", "index": index_or(c, i), "label": label(c)},
            }));
        }
    }

    if prompt == "BRAIN_LEECH.PAGES.SHARE_KNOWLEDGE.SELECTIONSCREENPROMPT" {
        // The engine offers Share Knowledge's pick-1-of-5 grid as its
        // choose-a-card screen with no skip (`PendingCardSelect`), the
        // CARD_REWARD the student saw in training, not as a card grant
        // (GH #1736).
        let mut rm = graft_screen_run(state, version, last_screen)?;
        let specs: Vec<Value> = cards.iter().map(|c| crate::graft::card_spec(c)).collect();
        sts2_wire::graft::open_card_offer(&mut rm, &Value::Array(specs), false, true, None)
            .map_err(GraftError::Value)?;
        let pick = ev
            .evaluate(&mut rm, &["CARD_REWARD"])
            .map_err(GraftError::Unavailable)?;
        let rows: Vec<Option<(Value, String)>> = (0..pick.probs.len())
            .map(|ci| {
                let a = first_action(&pick, ci);
                if py::get(&a, "action").as_str() != Some("pick_card_reward") {
                    return None;
                }
                let i = usize::try_from(py::int_of(py::get(&a, "idx"))).ok()?;
                let c = cards.get(i)?;
                Some((choose(index_or(c, i)), label(c)))
            })
            .collect();
        return advise(&pick, &rows);
    }
    if kind == "grant" {
        // The engine holds the earlier picks of a pick-N grant: its minimum
        // falls and its picked count rises as the screen fills (RUNCTX
        // `R_SEL_MIN`, `R_SEL_PICKED`; GH #1673).
        let count = usize::try_from(n).unwrap_or(usize::MAX);
        let mut rm = graft_screen_run(state, version, last_screen)?;
        let specs: Vec<Value> = all_cards
            .iter()
            .map(|c| crate::graft::card_spec(c))
            .collect();
        let lo_pick =
            usize::try_from(lo).map_err(|_| GraftError::Value(format!("min_select {lo} < 0")))?;
        sts2_wire::graft::queue_card_grant(&mut rm, &Value::Array(specs), count, lo_pick)
            .map_err(GraftError::Value)?;
        for (k, &p) in earlier.iter().enumerate() {
            // The engine's offer shrinks as it fills: an index among the
            // cards still on it.
            let at = p - earlier[..k].iter().filter(|&&q| q < p).count();
            let act = sts2_wire::emit::run_action_from_value(
                &json!({"action": "grant_deck_card", "idx": at}),
            )
            .map_err(|e| GraftError::Value(e.to_string()))?;
            let res = rm.take_action(act);
            if !matches!(res, sts2_core::ActionResult::Advanced) {
                return Err(unavailable(trf(
                    "the engine refused an earlier grant pick: {res}",
                    &[("res", &format!("{res:?}"))],
                )));
            }
        }
        ev.set_grant_picked(i64::try_from(earlier.len()).unwrap_or(0));
        let pick = ev
            .evaluate(&mut rm, &["CARD_GRANT"])
            .map_err(GraftError::Unavailable)?;
        let rows: Vec<Option<(Value, String)>> = (0..pick.probs.len())
            .map(|ci| {
                let a = first_action(&pick, ci);
                let action = py::str_of(py::get(&a, "action"));
                let idx = a.get("idx").and_then(Value::as_i64);
                match idx {
                    Some(i) if action == "grant_deck_card" && i < cards.len() as i64 => {
                        let c = py_index(&cards, i)?;
                        Some((choose(c.get("index").map_or(i, py::int_of)), label(c)))
                    }
                    _ if action.contains("confirm") || action.contains("skip") => {
                        Some((json!({"action": "skip"}), tr("take none").to_string()))
                    }
                    _ => None,
                }
            })
            .collect();
        return advise(&pick, &rows);
    }
    // Any other pick-N deck screen. The engine runs it as one pick per
    // queued entry, and a picked card leaves the later offers (enchanted,
    // upgraded, removed or transformed). So queue the whole screen, as the
    // engine queues it (`min(N, offer)` picks), and replay the earlier picks
    // in it before asking for the next (GH #1715).
    let n = if multi {
        n.min(i64::try_from(all_cards.len()).unwrap_or(i64::MAX))
    } else {
        n
    };
    let count = u32::try_from(n).unwrap_or(u32::MAX);

    let enchantment = if kind == "enchant" {
        Some(screen_enchantment(state).ok_or_else(|| {
            unavailable(tr("the screen does not say which enchantment it applies"))
        })?)
    } else {
        None
    };
    let (mut rm, expect) = if kind == "upgrade" && last_screen == Some("rest_site") {
        (smith_rm(state, version)?, "SMITH")
    } else {
        let mut rm = graft_screen_run(state, version, last_screen)?;
        let expect = match kind {
            "remove" => {
                if is_event(state, "AMALGAMATOR") {
                    // Amalgamator offers its Basic Strikes or Defends only,
                    // as the engine does (`removal_offer_order`, GH #1747).
                    let strikes = cards
                        .iter()
                        .all(|c| py::str_of(py::get(c, "id")).contains("STRIKE"));
                    rm.run_state.pending_removal_basic_tag = Some(if strikes {
                        sts2_core::cards::CardTag::Strike
                    } else {
                        sts2_core::cards::CardTag::Defend
                    });
                } else {
                    // Pael's Tooth removes its five from the upgradable cards
                    // only (`PaelsTooth.AfterObtained`, filter
                    // `c.IsUpgradable`): the screen's cards are those when
                    // the deck's removable ones are more (Spire Bench seed
                    // 180000416 offered the upgraded cards too).
                    let plain = rm.run_state.removal_offer_order().len();
                    rm.run_state.pending_removal_paels_tooth = true;
                    let upgradable = rm.run_state.removal_offer_order().len();
                    rm.run_state.pending_removal_paels_tooth =
                        plain != all_cards.len() && upgradable == all_cards.len();
                }
                rm.debug_queue_deck_removals(count);
                "CARD_REMOVE"
            }
            "upgrade" => {
                rm.debug_queue_deck_upgrades(count);
                "CARD_UPGRADE"
            }
            "enchant" => {
                let e = enchantment.as_deref().unwrap_or_default();
                // The amount reaches the deck tokens once an earlier pick is
                // replayed (GH #1715); a screen without it applies 1.
                let amount = state
                    .get("enchantment_amount")
                    .and_then(Value::as_i64)
                    .and_then(|a| i32::try_from(a).ok())
                    .unwrap_or(1);
                let (ty, second) = offered_card_types(&all_cards);
                sts2_wire::graft::queue_deck_enchant_of_types(
                    &mut rm,
                    e,
                    amount,
                    lo == 0,
                    count,
                    ty.as_deref(),
                    second.as_deref(),
                )
                .map_err(|_| unavailable(trf("the engine has no enchantment {e}", &[("e", &e)])))?;
                "CARD_ENCHANT"
            }
            _ => {
                // Wood Carvings transforms a Basic card only: the screen's
                // cards are the Basic ones when the deck's transformable
                // cards are more (GH #1680).
                let offered = |basic_only: bool| {
                    let mut probe = rm.clone();
                    probe.debug_queue_deck_transforms(1, basic_only);
                    probe.transform_eligible_deck_indices().len()
                };
                let basic_only =
                    offered(false) != all_cards.len() && offered(true) == all_cards.len();
                rm.debug_queue_deck_transforms(count, basic_only);
                "CARD_TRANSFORM"
            }
        };
        (rm, expect)
    };
    if multi && expect != "SMITH" {
        let verb = match expect {
            "CARD_REMOVE" => "remove_deck_card",
            "CARD_UPGRADE" => "upgrade_deck_card",
            "CARD_ENCHANT" => "enchant_deck_card",
            _ => "transform_deck_card",
        };
        for &p in &earlier {
            let want = all_cards.get(p).map(|c| wire_card_key(c)).ok_or_else(|| {
                unavailable(trf(
                    "an earlier pick {p} is not on the screen",
                    &[("p", &p)],
                ))
            })?;
            let act = (0..all_cards.len())
                .map(|idx| json!({"action": verb, "idx": idx}))
                .find(|a| picked_deck_card(&rm, a).is_some_and(|k| k == want))
                .ok_or_else(|| unavailable(tr("the engine does not offer an earlier pick")))?;
            let act = sts2_wire::emit::run_action_from_value(&act)
                .map_err(|e| GraftError::Value(e.to_string()))?;
            let res = rm.take_action(act);
            if !matches!(res, sts2_core::ActionResult::Advanced) {
                return Err(unavailable(trf(
                    "the engine refused an earlier pick: {res}",
                    &[("res", &format!("{res:?}"))],
                )));
            }
        }
    }
    let pick = ev
        .evaluate(&mut rm, &[expect])
        .map_err(GraftError::Unavailable)?;
    let rows: Vec<Option<(Value, String)>> = (0..pick.probs.len())
        .map(|ci| {
            let a = first_action(&pick, ci);
            if let Some(i) = picked_deck_card(&rm, &a).and_then(|k| find_card(&cards, &k)) {
                return Some((choose(index_or(cards[i], i)), label(cards[i])));
            }
            let action = py::str_of(py::get(&a, "action"));
            (action.contains("skip") || action.contains("decline"))
                .then(|| (json!({"action": "skip"}), tr("none").to_string()))
        })
        .collect();
    advise(&pick, &rows)
}

/// An engine deck card (identity `key`) by the name the run's wire deck
/// gives it, else its id's display name.
fn deck_card_name(state: &Value, key: &CardKey, names: &Names) -> String {
    let deck = dicts(py::get(state, "run_state"), "master_deck");
    match find_card(&deck, key) {
        Some(i) => names.card_name(deck[i]),
        None => names.display_name(&json!(key.0)) + if key.1 { "+" } else { "" },
    }
}

/// The shop as `sts2_wire::graft::open_shop` takes it: the bridge's
/// purchasable options, each kind in option order. The Fake Merchant's
/// shopfront (GH #1607, `event_name` FAKE_MERCHANT) opens as its event-room
/// shop: rolled prices, no removal.
fn shop_inventory(
    state: &Value,
    cards: &[&Value],
    relics: &[&Value],
    potions: &[&Value],
    remove_opt: Option<&Value>,
    removal_cost: &Value,
) -> Value {
    let item_id = |o: &Value| py::get(py::get(o, "item"), "id").clone();
    let price = |o: &Value| {
        let p = py::get(o, "price");
        if py::truthy(p) {
            py::int_of(p)
        } else {
            0
        }
    };
    json!({
        "cards": cards.iter().map(|o| json!({
            "id": item_id(o),
            "upgraded": py::truthy(py::get(py::get(o, "item"), "upgrade_level")),
            "price": price(o),
            "on_sale": false,
            "is_colorless": false,
            "available": true,
        })).collect::<Vec<_>>(),
        "relics": relics.iter().map(|o| json!({"id": item_id(o), "price": price(o), "available": true})).collect::<Vec<_>>(),
        "potions": potions.iter().map(|o| json!({"id": item_id(o), "price": price(o), "available": true})).collect::<Vec<_>>(),
        "removal_cost": if py::truthy(removal_cost) { py::int_of(removal_cost) } else { 0 },
        "removal_used": remove_opt.is_none(),
        // The Fake Merchant's shopfront (GH #1607): six fake relics in an
        // event room, the mod names it.
        "fake_merchant": py::get(state, "event_name").as_str() == Some("FAKE_MERCHANT"),
    })
}

/// The shop. Returns the advice and the deck card Instinct removes here
/// (the follow-up card screen executes it), when its pick is a removal.
///
/// # Errors
///
/// As [`card_reward`].
#[allow(clippy::too_many_lines)]
pub fn shop(
    ev: &mut dyn Evaluator,
    state: &Value,
    version: &str,
    names: &Names,
) -> Result<(Value, Option<CardKey>)> {
    let options = dicts(state, "options");
    if options.is_empty() {
        return Err(unavailable(tr("no shop options")));
    }
    // The Blitz merchant's wares are none of the kinds below, so Instinct
    // saw a shop whose only option was Leave and walked out on three free
    // picks. The advisor's Blitz rule answers it.
    let blitz = !py::get(state, "blitz_choices_remaining").is_null()
        || options
            .iter()
            .any(|o| py::get(o, "action").as_str() == Some("blitz_choice"));
    if blitz {
        return Err(unavailable(tr(
            "Instinct was not trained on the Blitz merchant",
        )));
    }
    let usable: Vec<&Value> = options
        .iter()
        .filter(|o| o.get("enabled").is_none_or(py::truthy))
        .copied()
        .collect();
    let of_kind = |k: &str| -> Vec<&Value> {
        usable
            .iter()
            .filter(|o| py::get(o, "action").as_str() == Some(k))
            .copied()
            .collect()
    };
    let (cards, relics, potions) = (
        of_kind("buy_card"),
        of_kind("buy_relic"),
        of_kind("buy_potion"),
    );
    let remove_opt = usable
        .iter()
        .find(|o| py::get(o, "action").as_str() == Some("remove_card"))
        .copied();
    let leave_opt = options
        .iter()
        .find(|o| py::get(o, "action").as_str() == Some("leave_shop"))
        .copied();
    let mut removal_cost = py::get(state, "purge_cost").clone();
    if removal_cost.is_null() {
        if let Some(r) = remove_opt {
            removal_cost = r.get("price").cloned().unwrap_or(json!(0));
        }
    }
    let inventory = shop_inventory(state, &cards, &relics, &potions, remove_opt, &removal_cost);
    let (mut rm, _notes) = graft_run(state, version)?;
    sts2_wire::graft::open_shop(&mut rm, &inventory)
        .map_err(|e| unavailable(trf("the engine cannot stock this shop ({e})", &[("e", &e)])))?;
    let pick = ev
        .evaluate(&mut rm, &["SHOP"])
        .map_err(GraftError::Unavailable)?;
    let deck = deck_keys(&rm);
    let mut removal_of: Vec<Option<CardKey>> = vec![None; pick.probs.len()];
    let rows: Vec<Option<(Value, String)>> = (0..pick.probs.len())
        .map(|ci| {
            let a = first_action(&pick, ci);
            let pick_from = |lst: &[&Value]| {
                let i = a.get("idx").and_then(Value::as_i64)?;
                let o = usize::try_from(i).ok().and_then(|i| lst.get(i))?;
                Some((
                    choose(py::int_of(py::get(o, "index"))),
                    names.shop_option_label(o),
                ))
            };
            match py::get(&a, "action").as_str() {
                Some("shop_buy_card") => pick_from(&cards),
                Some("shop_buy_relic") => pick_from(&relics),
                Some("shop_buy_potion") => pick_from(&potions),
                Some("shop_remove_card") if remove_opt.is_some() => {
                    let d = a.get("deck_idx").and_then(Value::as_i64)?;
                    let key = usize::try_from(d).ok().and_then(|d| deck.get(d))?.clone();
                    let text = trf(
                        "Remove {card}",
                        &[("card", &deck_card_name(state, &key, names))],
                    );
                    removal_of[ci] = Some(key);
                    Some((choose(py::int_of(py::get(remove_opt?, "index"))), text))
                }
                Some("leave_shop") if leave_opt.is_some() => Some((
                    choose(py::int_of(py::get(leave_opt?, "index"))),
                    tr("Leave shop").to_string(),
                )),
                _ => None,
            }
        })
        .collect();
    let advice = advise(&pick, &rows)?;
    // Carry the student's removal target to the follow-up card screen.
    let mut order: Vec<usize> = (0..pick.probs.len()).collect();
    order.sort_by(|&a, &b| pick.probs[b].total_cmp(&pick.probs[a]));
    let first_on_screen = order.into_iter().find(|&c| rows[c].is_some());
    let removal = first_on_screen.and_then(|c| removal_of[c].clone());
    Ok((advice, removal))
}

// ── events (Neow included) ──────────────────────────────────────────────

/// Event pages are reached by replaying earlier picks from page 0; bound the
/// search (the deepest live event is a few pages).
const EVENT_REPLAY_DEPTH: usize = 3;

/// A tail without its page number (`EXTRACT_CURRENT_PRIZE_2` ->
/// `..._PRIZE`): some engine events keep page 1's loc keys on every page.
fn loose(tail: &str) -> String {
    tail.trim_end_matches(|c: char| c.is_ascii_digit())
        .to_string()
}

fn event_infos(rm: &RunManager) -> Vec<Value> {
    match sts2_wire::emit::run_manager::event_option_infos_with(rm, false) {
        Value::Array(a) => a,
        _ => Vec::new(),
    }
}

/// The engine page's option names: the game's own loc key, else the
/// engine's option id.
fn engine_event_tails(rm: &RunManager) -> Vec<String> {
    event_infos(rm)
        .iter()
        .map(|i| {
            let t = norm_id(py::get(i, "loc_key"));
            if t.is_empty() {
                norm_id(py::get(i, "option_id"))
            } else {
                t
            }
        })
        .collect()
}

fn is_subsequence(want: &[String], have: &[String]) -> bool {
    let mut it = have.iter();
    want.iter().all(|w| it.any(|h| h == w))
}

fn choose_event(rm: &mut RunManager, idx: usize) -> bool {
    match sts2_wire::emit::run_action_from_value(&json!({"action": "event_choice", "idx": idx})) {
        Ok(act) => {
            let _ = rm.take_action(act);
            true
        }
        Err(_) => false,
    }
}

fn in_event(rm: &RunManager) -> bool {
    sts2_wire::emit::names::phase_str(rm.phase()) == "event"
}

/// How many placeholder seeds a seedless rebuild tries for an event page
/// whose options are rolled (Tinker Time's riders, the Colorful
/// Philosophers' characters).
const SEEDLESS_REROLLS: u32 = 64;

/// Placeholders tried for an event page whose buttons match the screen's
/// but whose rolled amounts do not (an amount rolls on ~20 values).
const SEEDLESS_VAR_REROLLS: u32 = 256;

/// Placeholder seeds a two-gold search tries: Jungle Maze's golds roll on
/// ~900 pairs, Sunken Treasury's on 976, so this finds the screen's pair
/// with near certainty.
const ROLLED_GOLDS_SEARCH: u32 = 20_000;

/// A placeholder seed whose dice roll the two golds the screen shows, for an
/// event whose golds depend only on the run's master seed (Jungle Maze,
/// Sunken Treasury), or `None` for another event or when none does. Each
/// try is two draws, not a rebuild.
fn rolled_golds_placeholder(page: &RunManager, options: &[&Value]) -> Option<String> {
    type Roll = fn(&sts2_core::run_rng_set::RunRngSet) -> (i32, i32);
    let events: [(&str, &str, &str, Roll); 2] = [
        (
            "JUNGLE_MAZE_ADVENTURE.",
            "SoloGold",
            "JoinForcesGold",
            sts2_content::events::act2_hive::jungle_maze_gold_for,
        ),
        (
            "SUNKEN_TREASURY.",
            "SmallChestGold",
            "LargeChestGold",
            sts2_content::events::shared_wave_b::sunken_treasury_gold_for,
        ),
    ];
    let (_, a, b, roll) = events.into_iter().find(|(prefix, ..)| {
        options
            .iter()
            .any(|o| py::str_of(py::get(o, "text_key")).starts_with(prefix))
    })?;
    let shown = |name: &str| {
        options
            .iter()
            .find_map(|o| py::get(o, "vars").get(name).and_then(Value::as_i64))
    };
    let want = (shown(a)?, shown(b)?);
    let version = page.run_state.game_version;
    (1..=ROLLED_GOLDS_SEARCH)
        .map(|n| format!("{}-jm{n}", crate::graft::SEEDLESS_SEED))
        .find(|seed| {
            let master = version.run_seed_hash(&sts2_core::canonicalize_seed(seed));
            let rng = sts2_core::run_rng_set::RunRngSet::new_for(version, master);
            let (x, y) = roll(&rng);
            (i64::from(x), i64::from(y)) == want
        })
}

/// A later page replayed on the engine's dice may have rolled the earlier
/// picks differently from the game (Tablet of Truth's random upgrade); the
/// screen's own deck, HP and gold are what the run holds now (GH #1689).
///
/// An earlier page's pick may also have queued a choice the game already
/// resolved on its own screen before showing this page (Endless Conveyor's
/// Jelly Liver transform, Suspicious Condiment's potion reward): the screen
/// is the event page, so nothing is pending. Left queued, the rebuilt run
/// stands at that choice (`CARD_TRANSFORM`) instead of the page, and
/// Instinct cannot answer.
fn restore_screen_run(page: &mut RunManager, state: &Value) -> Result<()> {
    page.debug_clear_pending_choices();
    let run = py::get(state, "run_state");
    if let Some(deck) = py::get(run, "master_deck").as_array() {
        let specs: Vec<Value> = deck
            .iter()
            .filter(|c| c.is_object())
            .map(crate::graft::card_spec)
            .collect();
        sts2_wire::graft::set_master_deck(page, &Value::Array(specs))
            .map_err(|e| unavailable(trf("the screen's deck: {e}", &[("e", &e)])))?;
    }
    let num = |k: &str| py::get(run, k).as_i64().and_then(|v| i32::try_from(v).ok());
    page.debug_set_player(num("hp"), num("max_hp"), num("gold"));
    // The belt too: an earlier page's potion (Endless Conveyor's Suspicious
    // Condiment) rolls on the engine's dice and waits as a reward there,
    // while the game's player already took it.
    if let Some(belt) = py::get(run, "potions").as_array() {
        // By registry `entry`, else `id`, as the run import reads them.
        let potions: Vec<String> = belt
            .iter()
            .filter(|p| p.is_object() && !py::truthy(py::get(p, "empty")))
            .filter_map(|p| {
                let named = |k: &str| {
                    py::get(p, k)
                        .as_str()
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                };
                let raw = named("entry").or_else(|| named("id"))?;
                sts2_content::names::potion_kind_from_str(raw).map(|k| k.python_name().to_owned())
            })
            .collect();
        sts2_wire::graft::set_potion_state(page, &potions, &[])
            .map_err(|e| unavailable(trf("the screen's potions: {e}", &[("e", &e)])))?;
    }
    Ok(())
}

/// How far the numbers the screen shows on its buttons (`vars`: the event's
/// DynamicVars, `Gold`, `HpLoss`, ...) are from the ones the engine's
/// matching buttons hold: the summed absolute difference, 0 when they agree.
/// A name only one side has is no evidence either way.
fn vars_distance(page: &RunManager, options: &[&Value]) -> i64 {
    let infos = event_infos(page);
    let slot_option = match_options(&engine_event_tails(page), options);
    infos
        .iter()
        .zip(&slot_option)
        .map(|(info, screen)| {
            let Some(screen) = screen else { return 0 };
            let (Some(have), Some(want)) = (
                py::get(info, "vars").as_object(),
                py::get(screen, "vars").as_object(),
            ) else {
                return 0;
            };
            want.iter()
                .map(
                    |(k, w)| match (have.get(k).and_then(Value::as_i64), w.as_i64()) {
                        (Some(h), Some(w)) => (h - w).abs(),
                        _ => 0,
                    },
                )
                .sum::<i64>()
        })
        .sum()
}

/// A loc key's page: `TRIAL.pages.NOBLE.options.GUILTY` ->
/// `TRIAL.PAGES.NOBLE`. Empty for a key with no `.options.` part.
fn page_of(key: &Value) -> String {
    let k = if py::truthy(key) {
        py::str_of(key)
    } else {
        String::new()
    };
    k.to_uppercase()
        .split_once(".OPTIONS.")
        .map_or_else(String::new, |(page, _)| page.to_owned())
}

/// The screen's buttons sit on the page the engine shows: events whose
/// pages share their button names (Trial's three trials all offer Guilty /
/// Innocent) are told apart by the page in the button's key (GH #1739).
fn pages_agree(page: &RunManager, options: &[&Value]) -> bool {
    let infos = event_infos(page);
    let have: BTreeSet<String> = infos
        .iter()
        .map(|i| page_of(py::get(i, "loc_key")))
        .filter(|p| !p.is_empty())
        .collect();
    if have.is_empty() {
        return labels_agree(&infos, options);
    }
    options
        .iter()
        .map(|o| page_of(py::get(o, "text_key")))
        .filter(|p| !p.is_empty())
        .all(|p| have.contains(&p))
        && names_agree(&infos, options)
}

/// A rolled button shares its page and its neighbours with any other roll
/// (Endless Conveyor's dish beside Leave): when some of the screen's button
/// names are the engine's, all of them must be. A page whose names differ
/// throughout (Colossal Flower keeps page 1's on later pages) is no evidence.
fn names_agree(infos: &[Value], options: &[&Value]) -> bool {
    let have: Vec<String> = infos
        .iter()
        .map(|i| loose(&norm_id(py::get(i, "loc_key"))))
        .collect();
    let shown: Vec<String> = options
        .iter()
        .map(|o| loose(&norm_id(py::get(o, "text_key"))))
        .collect();
    !shown.iter().any(|s| have.contains(s)) || is_subsequence(&shown, &have)
}

/// A page whose buttons have no game key (Doll Room's dolls, named after the
/// relic each slot rolled): when the screen's titles name the engine's
/// buttons, they must be the same buttons in the same order.
fn labels_agree(infos: &[Value], options: &[&Value]) -> bool {
    let squash = |v: &Value| -> String {
        py::str_of(v)
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .collect::<String>()
            .to_lowercase()
    };
    let named: Vec<String> = infos.iter().map(|i| squash(py::get(i, "label"))).collect();
    let shown: Vec<String> = options
        .iter()
        .map(|o| {
            let t = py::get(o, "title");
            squash(if py::truthy(t) {
                t
            } else {
                py::get(o, "label")
            })
        })
        .collect();
    !shown.iter().any(|s| named.contains(s)) || named == shown
}

fn vars_agree(page: &RunManager, options: &[&Value]) -> bool {
    vars_distance(page, options) == 0
}

/// `rm` (page 0) or a replayed copy showing the screen's options: the
/// screen's option names must appear among the page's, in order (the screen
/// lists only the buttons the game unlocked).
fn find_event_page(rm: &RunManager, want: &[String]) -> Option<RunManager> {
    let mut frontier = vec![rm.clone()];
    for _ in 0..=EVENT_REPLAY_DEPTH {
        let mut next = Vec::new();
        for cand in frontier {
            if !want.is_empty() && is_subsequence(want, &engine_event_tails(&cand)) {
                return Some(cand);
            }
            for i in 0..event_infos(&cand).len() {
                let mut probe = cand.clone();
                if choose_event(&mut probe, i) && in_event(&probe) {
                    next.push(probe);
                }
            }
        }
        frontier = next;
    }
    None
}

/// `rm` replayed through `trail` from page 0, if it shows the screen.
fn replay_trail(rm: &RunManager, trail: &[i64], want: &[String]) -> Option<RunManager> {
    if trail.is_empty() {
        return None;
    }
    let mut probe = rm.clone();
    for &idx in trail {
        if !choose_event(&mut probe, usize::try_from(idx).ok()?) || !in_event(&probe) {
            return None;
        }
    }
    let have: Vec<String> = engine_event_tails(&probe)
        .iter()
        .map(|t| loose(t))
        .collect();
    let want_loose: Vec<String> = want.iter().map(|t| loose(t)).collect();
    // The trail is the stronger evidence: some engine events keep page 1's
    // option names on later pages (Colossal Flower), so a page the trail
    // reaches with as many options as the screen is taken as the screen.
    (is_subsequence(&want_loose, &have) || have.len() == want.len()).then_some(probe)
}

/// Engine option i -> the screen's button for it (`None` when the screen
/// leaves it out, locked): by exact name, else by name without its page
/// number, else by position when no name is shared.
fn match_options<'a>(tails: &[String], options: &[&'a Value]) -> Vec<Option<&'a Value>> {
    let tail_of = |o: &Value| norm_id(py::get(o, "text_key"));
    let exact = |t: &str| options.iter().rev().find(|o| tail_of(o) == t).copied();
    let near = |t: &str| {
        options
            .iter()
            .rev()
            .find(|o| loose(&tail_of(o)) == t)
            .copied()
    };
    // Buttons that share a key (The Future of Potions offers one per potion,
    // all `…options.POTION`) pair up in order, each screen button once
    // (GH #1691): exact keys first, then loose ones; a key with no button
    // left falls back to the last one that matches.
    let mut used = vec![false; options.len()];
    let mut out: Vec<Option<&Value>> = vec![None; tails.len()];
    for pass in [false, true] {
        for (k, t) in tails.iter().enumerate() {
            if out[k].is_some() {
                continue;
            }
            let same = |o: &str| if pass { loose(o) == loose(t) } else { o == t };
            if let Some(i) = (0..options.len()).find(|&i| !used[i] && same(&tail_of(options[i]))) {
                used[i] = true;
                out[k] = Some(options[i]);
            }
        }
    }
    for (k, t) in tails.iter().enumerate() {
        if out[k].is_none() {
            out[k] = exact(t).or_else(|| near(&loose(t)));
        }
    }
    if out.iter().all(Option::is_none) && tails.len() == options.len() {
        return options.iter().map(|o| Some(*o)).collect();
    }
    out
}

/// This event's trail (`_event_track`): the engine option taken on each
/// earlier page, one per event visit keyed by seed, floor and event. When a
/// request shows another page than the one last answered, the option taken
/// there is the player's (`note_event_choice`) or, when the AI played it,
/// Instinct's own pick.
///
/// The track is plain JSON the host keeps between requests: `{key, trail,
/// pending, origin}`, `pending` = `{sig, wire_to_engine: [[wire, engine]],
/// best, human}`.
/// What tells one page of an event from the next: its buttons' full keys
/// (Tablet of Truth's DECIPHER_1..4 pages share their last segment, so the
/// short names saw one page and dropped every pick after the first, GH #1692)
/// and the run's HP, max HP, gold and deck size the last pick changed.
fn page_sig(state: &Value, options: &[&Value]) -> Value {
    let run = py::get(state, "run_state");
    json!({
        "keys": options.iter().map(|o| py::str_of(py::get(o, "text_key"))).collect::<Vec<_>>(),
        "run": [py::get(run, "hp"), py::get(run, "max_hp"), py::get(run, "gold"),
                py::list(run, "master_deck").len()],
    })
}

fn event_track(track: Option<&Value>, state: &Value, event_id: &str, sig: &Value) -> Value {
    let key = json!([
        py::get(state, "seed"),
        py::get(state, "floor"),
        norm_str(event_id)
    ]);
    let mut track = match track {
        Some(t) if py::eq(py::get(t, "key"), &key) => t.clone(),
        _ => json!({"key": key, "trail": [], "pending": null, "origin": state}),
    };
    let pending = track["pending"].clone();
    if !pending.is_null() && &pending["sig"] != sig {
        let chosen = if pending["human"].is_null() {
            &pending["best"]
        } else {
            &pending["human"]
        };
        if !chosen.is_null() {
            if let Some(trail) = track["trail"].as_array_mut() {
                trail.push(chosen.clone());
            }
        }
        track["pending"] = Value::Null;
    }
    track
}

/// A `human_action` on an event screen: the page's taken option.
#[must_use]
pub fn note_event_choice(track: &Value, msg: &Value) -> Value {
    let mut track = track.clone();
    let Some(index) = py::get(py::get(msg, "choice"), "index").as_i64() else {
        return track;
    };
    if track.get("pending").is_none_or(Value::is_null) {
        return track;
    }
    let engine = py::list(&track["pending"], "wire_to_engine")
        .iter()
        .rev()
        .find(|p| p[0].as_i64() == Some(index))
        .map_or(Value::Null, |p| p[1].clone());
    track["pending"]["human"] = engine;
    track
}

/// An event page (Neow and the Ancients' relic offers included).
///
/// `track` is the host's event trail from the last request; the returned one
/// replaces it, whether or not the advice succeeds.
///
/// # Errors
///
/// As [`card_reward`].
#[allow(clippy::too_many_lines)]
pub fn event(
    ev: &mut dyn Evaluator,
    state: &Value,
    version: &str,
    track: Option<&Value>,
) -> (Result<Value>, Option<Value>) {
    let options = withhold_lethal_options(dicts(state, "options"));
    if options.is_empty() {
        return (Err(unavailable(tr("no event options"))), track.cloned());
    }
    let enabled: Vec<&Value> = options
        .iter()
        .filter(|o| o.get("enabled").is_none_or(py::truthy))
        .copied()
        .collect();
    let event_id = py::get(options[0], "event_id");
    let event_id = if py::truthy(event_id) {
        py::str_of(event_id)
    } else {
        String::new()
    };
    if enabled.len() == 1 {
        let advice = crate::instinct::lone_option(enabled[0]);
        return (
            Ok(advice),
            lone_pick_track(track, state, &event_id, &options, enabled[0]),
        );
    }
    // A button the game shows locked under its own key (`…_LOCKED`, Self-Help
    // Book's unaffordable read) has no engine option: the engine leaves it out.
    let options: Vec<&Value> = options
        .into_iter()
        .filter(|o| !py::str_of(py::get(o, "text_key")).ends_with("_LOCKED"))
        .collect();
    let want: Vec<String> = options
        .iter()
        .map(|o| norm_id(py::get(o, "text_key")))
        .collect();
    if norm_str(&event_id) == "NEOW" {
        return (neow(ev, state, version, &options), track.cloned());
    }
    // An Ancient's relic offer is the engine's boss-relic pick; another
    // event whose buttons name relics (Tea Master's teas) stays an event
    // (GH #1681).
    let ancient = {
        use sts2_core::run_room_gen::AncientId as A;
        let id = norm_str(&event_id);
        id.is_empty()
            || [
                A::Orobas,
                A::Pael,
                A::Tezcatara,
                A::Nonupeipe,
                A::Tanx,
                A::Vakuu,
                A::Darv,
            ]
            .iter()
            .any(|a| norm_str(a.entry()) == id)
    };
    if ancient && enabled.iter().all(|o| py::truthy(py::get(o, "relic_id"))) {
        return (
            ancient_relic_choice(ev, state, version, &options),
            track.cloned(),
        );
    }
    // A later page is replayed from the event's FIRST page: the envelope of
    // a later one already carries the earlier picks' HP / gold / cards, and
    // replaying them on top of it would count them twice.
    let sig = page_sig(state, &options);
    let mut track = event_track(track, state, &event_id, &sig);
    let out = (|| -> Result<Value> {
        let trail: Vec<i64> = py::list(&track, "trail")
            .iter()
            .filter_map(Value::as_i64)
            .collect();
        let page_on = |origin: &Value| -> Result<Option<RunManager>> {
            let (mut rm, _notes) = graft_run(origin, version)?;
            sts2_wire::graft::enter_event(&mut rm, &event_id).map_err(|_| {
                unavailable(trf(
                    "the engine has no event {event_id}",
                    &[("event_id", &event_id)],
                ))
            })?;
            Ok(replay_trail(&rm, &trail, &want).or_else(|| find_event_page(&rm, &want)))
        };
        let mut page = page_on(&track["origin"])?;
        if page.is_none() && crate::graft::seedless(&track["origin"]) {
            // No seed, so the event rolls its options on placeholder dice.
            // Other placeholders until one rolls the screen's; the event's
            // later pages replay on the same one.
            for n in 1..=SEEDLESS_REROLLS {
                let mut origin = track["origin"].clone();
                origin["seedless_seed"] = json!(format!("{}-{n}", crate::graft::SEEDLESS_SEED));
                if let Some(p) = page_on(&origin)? {
                    track["origin"] = origin;
                    page = Some(p);
                    break;
                }
            }
        }
        if crate::graft::seedless(&track["origin"])
            && page.as_ref().is_some_and(|p| !pages_agree(p, &options))
        {
            // The placeholder dice rolled a page with the screen's buttons
            // but not the screen's page (Trial's three trials all offer
            // Guilty / Innocent; GH #1739): other placeholders until one
            // rolls it. None does: the page found stands.
            for n in 1..=SEEDLESS_REROLLS {
                let mut origin = track["origin"].clone();
                origin["seedless_seed"] = json!(format!("{}-p{n}", crate::graft::SEEDLESS_SEED));
                if let Some(p) = page_on(&origin)?.filter(|p| pages_agree(p, &options)) {
                    track["origin"] = origin;
                    page = Some(p);
                    break;
                }
            }
        }
        if crate::graft::seedless(&track["origin"])
            && trail.is_empty()
            && page.as_ref().is_some_and(|p| !vars_agree(p, &options))
        {
            // The placeholder dice rolled the screen's buttons but not its
            // amounts (an event's `CalculateVars`: WhisperingHollow's gold,
            // GH #1671); the student reads the amounts, so other
            // placeholders until one rolls those too. Two amounts roll on ~900
            // pairs (Jungle Maze's golds, GH #1686), so failing an exact roll
            // the closest one stands.
            let mut best = page
                .as_ref()
                .map_or(i64::MAX, |p| vars_distance(p, &options));
            // Jungle Maze's and Sunken Treasury's pairs roll off the event's
            // named stream alone, so their placeholder is found without a
            // rebuild per try.
            if let Some(found) = page
                .as_ref()
                .and_then(|p| rolled_golds_placeholder(p, &options))
            {
                let mut origin = track["origin"].clone();
                origin["seedless_seed"] = json!(found);
                if let Some(p) = page_on(&origin)? {
                    let d = vars_distance(&p, &options);
                    if d < best {
                        best = d;
                        track["origin"] = origin;
                        page = Some(p);
                    }
                }
            }
            for n in 1..=SEEDLESS_VAR_REROLLS {
                let mut origin = track["origin"].clone();
                origin["seedless_seed"] = json!(format!("{}-v{n}", crate::graft::SEEDLESS_SEED));
                let Some(p) = page_on(&origin)? else { continue };
                let d = vars_distance(&p, &options);
                if d < best {
                    best = d;
                    track["origin"] = origin;
                    page = Some(p);
                    if d == 0 {
                        break;
                    }
                }
            }
        }
        let mut page =
            page.ok_or_else(|| unavailable(tr("the rebuilt event does not show this page")))?;
        if !trail.is_empty() {
            restore_screen_run(&mut page, state)?;
            // Endless Conveyor's dish after a grab: the placeholder replay of
            // a long visit runs out of gold or rolls other dishes, and the
            // page found stands with another dish than the screen's (Spire
            // Bench seeds 180000706, 180000544, 180001732). The screen names
            // the dish; put it on the belt.
            if norm_str(&event_id) == "ENDLESSCONVEYOR" {
                if let Some(dish) = options
                    .iter()
                    .map(|o| py::str_of(py::get(o, "text_key")))
                    .find(|k| k.contains(".pages.ALL.options."))
                {
                    sts2_content::events::act2_hive::endless_conveyor_graft_dish(
                        &mut page.run_state,
                        trail.len(),
                        &dish,
                    );
                }
            }
        }
        let slot_option = match_options(&engine_event_tails(&page), &options);
        let pick = ev
            .evaluate(&mut page, &["EVENT"])
            .map_err(GraftError::Unavailable)?;
        let advice = advise(&pick, &slot_rows(&pick, &slot_option))?;
        // What this page's buttons are in the engine, and Instinct's own
        // pick: the next page of this event replays whichever was taken.
        let wire_to_engine: Vec<Value> = slot_option
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.map(|o| json!([index_or(o, i), i])))
            .collect();
        let best = advice["best"]
            .get("index")
            .and_then(Value::as_i64)
            .map_or(Value::Null, |b| {
                wire_to_engine
                    .iter()
                    .rev()
                    .find(|p| p[0].as_i64() == Some(b))
                    .map_or(Value::Null, |p| p[1].clone())
            });
        track["pending"] =
            json!({"sig": sig, "wire_to_engine": wire_to_engine, "best": best, "human": null});
        Ok(advice)
    })();
    (out, Some(track))
}

/// The track after a lone-button page, whose one button is taken without
/// asking: the page's pick like any answered page's, so a later page's
/// trail, replayed from the event's first page, holds it (Tinker Time's
/// first page). The button's engine option is its place among the buttons
/// the engine has (a `…_LOCKED` one it leaves out). Neow and the Ancients
/// keep no trail.
fn lone_pick_track(
    track: Option<&Value>,
    state: &Value,
    event_id: &str,
    options: &[&Value],
    lone: &Value,
) -> Option<Value> {
    let id = norm_str(event_id);
    if id.is_empty() || id == "NEOW" {
        return track.cloned();
    }
    let kept: Vec<&Value> = options
        .iter()
        .filter(|o| !py::str_of(py::get(o, "text_key")).ends_with("_LOCKED"))
        .copied()
        .collect();
    let Some(engine) = kept.iter().position(|o| std::ptr::eq(*o, lone)) else {
        return track.cloned();
    };
    let sig = page_sig(state, &kept);
    let mut track = event_track(track, state, event_id, &sig);
    track["pending"] = json!({"sig": sig, "wire_to_engine": [[index_or(lone, engine), engine]],
                              "best": engine, "human": null});
    Some(track)
}

/// The event page's buttons minus the ones the world flags
/// `will_kill_player`, unless no enabled button survives: the game's own
/// `AutoSlay.EventRoomHandler` rule, and the PPO wire's (`option_is_lethal`).
/// Instinct's engine run (agent mode) does not offer Trial's Double Down
/// (`ThatWillKillPlayerIf(_ => true)`, GH #1605), so the screen's page has to
/// drop it too, or the rebuilt page would not match the screen.
fn withhold_lethal_options(options: Vec<&Value>) -> Vec<&Value> {
    let enabled = |o: &Value| o.get("enabled").is_none_or(py::truthy);
    let lethal = |o: &Value| o.get("will_kill_player").and_then(Value::as_bool) == Some(true);
    if !options.iter().any(|o| enabled(o) && !lethal(o)) {
        return options;
    }
    options
        .into_iter()
        .filter(|o| !(enabled(o) && lethal(o)))
        .collect()
}

/// The student's classes over an event page's engine options.
fn slot_rows(pick: &Pick, slot_option: &[Option<&Value>]) -> Vec<Option<(Value, String)>> {
    (0..pick.probs.len())
        .map(|ci| {
            let i = first_action(pick, ci).get("idx").and_then(Value::as_i64)?;
            let o = usize::try_from(i)
                .ok()
                .and_then(|i| slot_option.get(i))
                .copied()
                .flatten()?;
            if !o.get("enabled").is_none_or(py::truthy) {
                return None;
            }
            let l = py::get(o, "label");
            Some((
                choose(index_or(o, i as usize)),
                if py::truthy(l) {
                    py::str_of(l)
                } else {
                    "?".into()
                },
            ))
        })
        .collect()
}

/// Neow: the run as Neow deals it (agent mode, on the live build), its
/// slots (`positive_1`, `cursed`) matched to the screen's relics.
fn neow(ev: &mut dyn Evaluator, state: &Value, version: &str, options: &[&Value]) -> Result<Value> {
    let rs = py::get(state, "run_state");
    let asc = py::get(rs, "ascension");
    let asc = if py::truthy(asc) { py::int_of(asc) } else { 0 };
    // A seedless request (Spire Bench) cannot roll the act-1 variant and its
    // boss from the seed: the screen names them (`act1`, the map's or the
    // context's boss). Rolled off a placeholder seed, the student saw another
    // boss than the game's (GH #1846).
    let seedless = crate::graft::seedless(state);
    let seed = if seedless {
        crate::graft::SEEDLESS_SEED.to_owned()
    } else {
        py::str_of(py::get(state, "seed"))
    };
    let act1 = if seedless {
        state.get("act1").and_then(Value::as_str).unwrap_or("roll")
    } else {
        "roll"
    };
    // The run's own character: its starter deck, relic and Neow pool (an
    // Ironclad rebuild showed every other character's Neow an Ironclad run).
    let character =
        crate::headless::character_display_name(py::get(rs, "character")).unwrap_or("Ironclad");
    let (mut rm, _) = sts2_wire::run_builder::build_run(&sts2_wire::run_builder::RunOptions {
        game_version: Some(version),
        string_seed: Some(&seed),
        character,
        ascension_level: u8::try_from(asc)
            .map_err(|_| GraftError::Value(format!("ascension {asc} out of range")))?,
        neow_start: true,
        mode: Some("agent"),
        interactive_potion_swap: true,
        act1,
        ..sts2_wire::run_builder::RunOptions::default()
    })
    .map_err(GraftError::Value)?;
    if seedless {
        let map = py::get(state, "map");
        let boss = if py::truthy(py::get(py::get(map, "boss"), "id")) {
            py::get(py::get(map, "boss"), "id")
        } else {
            py::get(py::get(py::get(state, "context"), "boss"), "id")
        };
        if let Some(b) = crate::graft::boss_encounter(boss)? {
            if let Some(rooms) = rm.run_state.generated_rooms.first_mut() {
                rooms.boss_id = b;
            }
        }
    }
    if crate::graft::is_coop(state) {
        rm.debug_use_coop_maps()
            .map_err(|e| GraftError::Value(format!("debug_use_coop_maps: {e}")))?;
    }
    // `{norm(relic_id): option}`, the last option for a relic winning.
    let with_relic: Vec<&Value> = options
        .iter()
        .filter(|o| py::truthy(py::get(o, "relic_id")))
        .copied()
        .collect();
    let mut relic_keys: Vec<String> = Vec::new();
    for o in &with_relic {
        let k = norm_id(py::get(o, "relic_id"));
        if !relic_keys.contains(&k) {
            relic_keys.push(k);
        }
    }
    let by_relic = |r: &str| {
        with_relic
            .iter()
            .rev()
            .find(|o| norm_id(py::get(o, "relic_id")) == r)
            .copied()
    };
    let offers = |rm: &RunManager| -> Vec<String> {
        sts2_wire::emit::run_manager::neow_offers(rm)
            .map(|(a, b, c)| vec![norm_str(&a), norm_str(&b), norm_str(&c)])
            .unwrap_or_default()
    };
    let mut slot_option: Vec<Option<&Value>> = offers(&rm).iter().map(|r| by_relic(r)).collect();
    let relic_ids: Vec<String> = options
        .iter()
        .map(|o| {
            let r = py::get(o, "relic_id");
            if py::truthy(r) {
                py::str_of(r)
            } else {
                String::new()
            }
        })
        .collect();
    let matched = |s: &[Option<&Value>]| s.iter().filter(|o| o.is_some()).count();
    // The screen's order is the game's: the two shuffled positives, then the
    // curse (`Neow.GenerateInitialOptions`, decomp `Neow.cs:281-283`). The
    // student reads each option's slot (X_POS), so the same three relics in
    // another order are a different screen: a seedless rebuild's placeholder
    // seed rolled Spire Bench seed 180001084's three in another order.
    let screen_order: Vec<String> = relic_ids.iter().map(|r| norm_str(r)).collect();
    if (matched(&slot_option) != relic_keys.len() || offers(&rm) != screen_order)
        && options.len() == 3
        && relic_ids.iter().all(|r| !r.is_empty())
    {
        // The engine rolls single player's Neow. Co-op seeds each player's
        // with their slot and keeps Massive Scroll in the pool, so the roll
        // differs there, and a seedless request rolls a placeholder seed's:
        // stand on the screen's own three offers, in its order.
        sts2_wire::graft::set_neow_offers(&mut rm, &relic_ids[0], &relic_ids[1], &relic_ids[2])
            .map_err(|e| {
                unavailable(trf(
                    "the engine has no Neow offer like this ({e})",
                    &[("e", &e)],
                ))
            })?;
        slot_option = offers(&rm).iter().map(|r| by_relic(r)).collect();
    }
    if matched(&slot_option) != relic_keys.len() || relic_keys.is_empty() {
        return Err(unavailable(tr(
            "the rebuilt Neow offers other relics than the screen",
        )));
    }
    let pick = ev
        .evaluate(&mut rm, &["NEOW"])
        .map_err(GraftError::Unavailable)?;
    advise(&pick, &slot_rows(&pick, &slot_option))
}

// ── the map ──────────────────────────────────────────────────────────────

/// A map node named the way it looks on screen (`Rest Site`).
fn node_label(node: &Value) -> String {
    sts2_runai::ppo::wire::map_labels(std::slice::from_ref(node))
        .pop()
        .map(|l| map_label(&l))
        .unwrap_or_default()
}

/// The wire `advice` for `map_select` (`map_advice`): each class is a
/// `move` to a column. The engine's move names the child by its column; the
/// mod's `nodes[]` name the same children with their `index` (what the
/// takeover clicks) and their coordinate (where the HUD pins the score).
///
/// # Errors
///
/// The student's moves do not match the screen's nodes.
pub fn map_advice(pick: &Pick, state: &Value) -> Result<Value> {
    let nodes = dicts(state, "nodes");
    // `{int(col): node}` over the nodes with an integer column, last wins.
    let by_col = |col: i64| {
        nodes
            .iter()
            .rev()
            .find(|n| py::get(n, "col").as_i64() == Some(col))
            .copied()
    };
    let mut options: Vec<Value> = Vec::new();
    for cls in py::list(&pick.decision, "classes") {
        let ci = py::int_of(py::get(cls, "class"));
        let action = py::as_list(py::get(cls, "actions"))
            .first()
            .filter(|a| py::truthy(a))
            .cloned()
            .unwrap_or_else(|| json!({}));
        if py::get(&action, "action").as_str() != Some("move") {
            return Err(unavailable(trf(
                "unexpected map move {action}",
                &[("action", &py::repr(&action))],
            )));
        }
        let col = action.get("col").map_or(-1, py::int_of);
        let node = by_col(col).ok_or_else(|| {
            unavailable(tr("Instinct's map moves do not match the screen's nodes"))
        })?;
        let p = usize::try_from(ci)
            .ok()
            .and_then(|c| pick.probs.get(c))
            .copied()
            .unwrap_or(0.0);
        let p = py::round(p, 4);
        let position = nodes.iter().position(|n| *n == node).unwrap_or(0);
        options.push(json!({
            "label": node_label(node),
            "visit_share": p,
            "prob": p,
            "is_best": usize::try_from(ci).ok() == Some(pick.best),
            "index": index_or(node, position),
            "action": "choose",
            "col": py::int_of(py::get(node, "col")),
            "row": py::int_of(py::get(node, "row")),
        }));
    }
    if options.len() != nodes.len() {
        return Err(unavailable(trf(
            "Instinct sees {n} paths, the map shows {m}",
            &[("n", &options.len()), ("m", &nodes.len())],
        )));
    }
    let best = options
        .iter()
        .find(|o| o["is_best"] == json!(true))
        .cloned()
        .ok_or_else(|| unavailable(tr("Instinct's best move is not on the map")))?;
    let note = crate::instinct::note_for(pick, best["prob"].as_f64().unwrap_or(0.0));
    options.sort_by(|a, b| {
        b["prob"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&a["prob"].as_f64().unwrap_or(0.0))
    });
    Ok(json!({
        "note": note,
        "options": options,
        "best": {"action": "choose", "index": best["index"], "label": best["label"]},
    }))
}

/// `advice.path` for Instinct: the student's own MAP choice, hop by hop.
///
/// The twin of the search's route projection, same schema and HP
/// projection: from the chosen node, follow single-child chains, and at
/// every fork park a clone of the grafted run ON the hop's node (the
/// projected HP set) and ask the student for its MAP decision there.
fn instinct_route(
    ev: &mut dyn Evaluator,
    state: &Value,
    rm: &RunManager,
    chosen: crate::route::Coord,
) -> Vec<Value> {
    let graph = crate::route::route_graph(state);
    if graph.is_empty() {
        return Vec::new();
    }
    let rs = py::get(state, "run_state");
    let num = |v: &Value| {
        if py::truthy(v) {
            v.as_f64().unwrap_or_else(|| py::int_of(v) as f64)
        } else {
            0.0
        }
    };
    let (hp, max_hp) = (num(py::get(rs, "hp")), num(py::get(rs, "max_hp")));
    let gold = {
        let g = py::get(rs, "gold");
        if py::truthy(g) {
            py::int_of(g)
        } else {
            0
        }
    };
    let act = {
        let a = py::get(state, "act");
        if py::truthy(a) {
            py::int_of(a)
        } else {
            1
        }
    };
    let floor = {
        let f = py::get(state, "floor");
        if py::truthy(f) {
            py::int_of(f)
        } else {
            0
        }
    };
    let base_total = sts2_core::runstate::engine_total_floor(floor, act);
    let base_visited: Vec<crate::route::Coord> = py::list(py::get(state, "map"), "visited")
        .iter()
        .filter_map(|v| Some((py::get(v, "col").as_i64()?, py::get(v, "row").as_i64()?)))
        .collect();
    let clamp = |v: i64| i32::try_from(v).unwrap_or(if v < 0 { i32::MIN } else { i32::MAX });
    crate::route::walk(
        &graph,
        chosen,
        act,
        hp,
        max_hp,
        |hop, cur, children, hp, walked| {
            let mut hop_rm = rm.clone();
            let visited: Vec<sts2_core::map::MapCoord> = base_visited
                .iter()
                .chain(walked)
                .map(|&(c, r)| sts2_core::map::MapCoord::new(clamp(c), clamp(r)))
                .collect();
            let at = sts2_core::map::MapCoord::new(clamp(cur.0), clamp(cur.1));
            let act8 = u8::try_from(act).ok()?;
            hop_rm
                .debug_place_on_map(
                    act8,
                    at,
                    visited,
                    clamp(cur.1 + 1),
                    clamp(base_total + hop as i64 + 1),
                )
                .ok()?;
            let hp_now = crate::route::round_half_even(hp).max(1);
            hop_rm.debug_set_player(
                Some(clamp(hp_now)),
                Some(clamp(crate::route::round_half_even(max_hp))),
                Some(clamp(gold)),
            );
            let pick = ev.evaluate(&mut hop_rm, &["MAP"]).ok()?;
            let action = first_action(&pick, pick.best);
            let col = py::get(&action, "col");
            children.iter().find(|c| py::eq(&json!(c.0), col)).copied()
        },
    )
}

/// The map screen (`_advise_map`); `state` already carries the session's
/// cached act graph (`realgame_ppo.apply_session_context`).
///
/// # Errors
///
/// As [`card_reward`].
pub fn map(ev: &mut dyn Evaluator, state: &Value, version: &str) -> Result<Value> {
    let nodes = dicts(state, "nodes");
    let (mut rm, notes) = match graft_run(state, version) {
        Ok((rm, notes)) => (Some(rm), notes),
        // A corridor still has its one answer, just no route.
        Err(GraftError::Unavailable(_)) if nodes.len() == 1 => (None, Vec::new()),
        Err(e) => return Err(e),
    };
    if notes.iter().any(|n| n == crate::graft::COOP_MAP_APPROX) {
        // Co-op runs are rebuilt on co-op sized maps; this one still differs
        // from the screen, so its paths are not real.
        if nodes.len() != 1 {
            return Err(unavailable(tr(
                "the rebuilt co-op map differs from the screen's",
            )));
        }
        rm = None;
    }
    let mut advice = if let [node] = nodes.as_slice() {
        // One way on (an act's entrance, a corridor): no decision, but still
        // the route from it.
        let mut labelled = (*node).clone();
        labelled["label"] = json!(node_label(node));
        let mut advice = crate::instinct::lone_option(&labelled);
        if let Some(opts) = advice["options"].as_array_mut() {
            for o in opts {
                o["col"] = json!(py::int_of(py::get(node, "col")));
                o["row"] = json!(py::int_of(py::get(node, "row")));
            }
        }
        advice
    } else {
        let rm = rm
            .as_mut()
            .ok_or_else(|| unavailable(tr("no run to rebuild")))?;
        let pick = ev.evaluate(rm, &["MAP"]).map_err(GraftError::Unavailable)?;
        map_advice(&pick, state)?
    };
    let best = py::list(&advice, "options")
        .iter()
        .find(|o| o["is_best"] == json!(true))
        .cloned();
    if let (Some(best), Some(rm)) = (best, rm.as_ref()) {
        let chosen = (py::int_of(&best["col"]), py::int_of(&best["row"]));
        let path = sts2_e2e::tokens::without_row_dump(|| instinct_route(ev, state, rm, chosen));
        if !path.is_empty() {
            let note = crate::route::route_note(&path);
            advice["path"] = Value::Array(path);
            if let Some(n) = advice["note"].as_str() {
                advice["note"] = json!(format!("{n}{note}"));
            }
        }
    }
    Ok(advice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_deck_card_screen_stands_in_the_live_room() {
        let live = |r: &str| json!({ "current_room": r });
        // The mod names the room: it wins over the last screen answered (a
        // grant from a fight's relic after a card reward).
        assert_eq!(
            screen_room(&live("Monster"), Some("card_reward")),
            Some("monster")
        );
        assert_eq!(screen_room(&live("Event"), Some("shop")), Some("event"));
        assert_eq!(screen_room(&live("RestSite"), None), Some("rest_site"));
        assert_eq!(screen_room(&live("Map"), Some("event")), None);
        // A mod without the field: the last screen, as before.
        assert_eq!(screen_room(&json!({}), Some("event")), Some("event"));
        assert_eq!(screen_room(&json!({}), Some("combat_action")), None);
        assert_eq!(screen_room(&json!({}), None), None);
    }

    #[test]
    fn a_lethal_event_button_is_withheld_while_another_survives() {
        let accept = json!({"index": 0, "text_key": "TRIAL.pages.REJECT.options.ACCEPT"});
        let double_down = json!({"index": 1, "will_kill_player": true,
            "text_key": "TRIAL.pages.REJECT.options.DOUBLE_DOWN"});
        let locked = json!({"index": 2, "enabled": false});
        assert_eq!(
            withhold_lethal_options(vec![&accept, &double_down, &locked]),
            vec![&accept, &locked]
        );
        // Every enabled button lethal: the pick stays the policy's.
        assert_eq!(
            withhold_lethal_options(vec![&double_down, &locked]),
            vec![&double_down, &locked]
        );
    }

    /// With Miniature Tent the game takes a used rest option off the screen
    /// and keeps the others. The second screen of the visit is the engine's
    /// site with that option taken, not a fresh one (Spire Bench seeds
    /// 170000054, 170000258).
    #[test]
    fn a_rest_option_gone_from_the_screen_was_taken_this_visit() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let mut msg = seedless_message(&screen_map(&reference_run(1)), 1);
        msg["type"] = json!("rest_site");
        msg["run_state"]["relics"] = json!([{"id": "BURNING_BLOOD"}, {"id": "MINIATURE_TENT"}]);
        msg["run_state"]["master_deck"] =
            json!([{"id": "BASH", "upgraded": 0}, {"id": "ANGER", "upgraded": 0}]);
        struct TakenProbe(Option<Vec<String>>);
        impl Evaluator for TakenProbe {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                let taken = &rm.run_state.rest_site_taken;
                self.0 = Some(taken.iter().map(|o| o.option_id().to_owned()).collect());
                Err("probe".into())
            }
        }
        let seen = |options: Value| {
            let mut msg = msg.clone();
            msg["options"] = options;
            let mut probe = TakenProbe(None);
            let _ = rest_site(&mut probe, &msg, "v0.111.0");
            probe.0.expect("the student was asked")
        };
        let both = json!([{"index": 0, "id": "HEAL"}, {"index": 1, "id": "SMITH"}]);
        assert!(seen(both).is_empty());
        assert_eq!(seen(json!([{"index": 0, "id": "HEAL"}])), ["SMITH"]);
    }

    /// GH #1833: the student's leave on a Miniature Tent visit's second
    /// screen is the screen's proceed, not the next option down.
    #[test]
    fn a_tent_visits_leave_is_answered_as_leave() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let mut msg = seedless_message(&screen_map(&reference_run(1)), 1);
        msg["type"] = json!("rest_site");
        msg["run_state"]["relics"] = json!([{"id": "BURNING_BLOOD"}, {"id": "MINIATURE_TENT"}]);
        msg["run_state"]["master_deck"] =
            json!([{"id": "BASH", "upgraded": 0}, {"id": "ANGER", "upgraded": 0}]);
        msg["options"] = json!([{"index": 0, "id": "HEAL"}]);
        msg["can_leave"] = json!(true);
        struct Leave;
        impl Evaluator for Leave {
            fn evaluate(
                &mut self,
                _: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                Ok(Pick {
                    decision: json!({"classes": [
                        {"actions": [{"action": "rest_option", "idx": 0}]},
                        {"actions": [{"action": "leave_shop"}]},
                    ]}),
                    probs: vec![0.2, 0.8],
                    best: 1,
                    ms: 0.0,
                })
            }
        }
        let advice = rest_site(&mut Leave, &msg, "v0.111.0").expect("answered");
        assert_eq!(advice["best"]["action"], json!("leave"));
    }

    /// GH #1846: a seedless Neow (Spire Bench) shows the student the act-1
    /// boss the screen names, not the one the placeholder seed rolls.
    #[test]
    fn a_seedless_neow_shows_the_screens_boss() {
        struct Boss(Option<&'static str>);
        impl Evaluator for Boss {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = rm.run_state.generated_rooms.first().map(|r| r.boss_id);
                Err("probe".into())
            }
        }
        let options = [
            json!({"index": 0, "event_id": "NEOW", "relic_id": "SCROLL_BOXES"}),
            json!({"index": 1, "event_id": "NEOW", "relic_id": "WINGED_BOOTS"}),
            json!({"index": 2, "event_id": "NEOW", "relic_id": "PRECARIOUS_SHEARS"}),
        ];
        let opts: Vec<&Value> = options.iter().collect();
        for (act1, boss) in [
            ("underdocks", "SOUL_FYSH_BOSS"),
            ("overgrowth", "VANTOM_BOSS"),
        ] {
            let state = json!({
                "seedless": true,
                "act1": act1,
                "run_state": {"ascension": 10},
                "context": {"boss": {"id": boss}},
            });
            let mut probe = Boss(None);
            let _ = neow(&mut probe, &state, "v0.111.0", &opts);
            assert_eq!(probe.0, Some(boss), "{act1}");
        }
    }

    /// A seedless Neow whose placeholder seed rolls the screen's three relics
    /// in another order (SEEDLESS rolls Phial Holster, Neow's Talisman, Leafy
    /// Poultice) offers them in the screen's order: the student reads each
    /// option's slot (Spire Bench seed 180001084).
    #[test]
    fn a_seedless_neow_keeps_the_screens_option_order() {
        struct Offers(Option<(String, String, String)>);
        impl Evaluator for Offers {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = sts2_wire::emit::run_manager::neow_offers(rm);
                Err("probe".into())
            }
        }
        let options = [
            json!({"index": 0, "event_id": "NEOW", "relic_id": "NEOWS_TALISMAN"}),
            json!({"index": 1, "event_id": "NEOW", "relic_id": "PHIAL_HOLSTER"}),
            json!({"index": 2, "event_id": "NEOW", "relic_id": "LEAFY_POULTICE"}),
        ];
        let opts: Vec<&Value> = options.iter().collect();
        let state = json!({
            "seedless": true,
            "act1": "overgrowth",
            "run_state": {"ascension": 10},
            "context": {"boss": {"id": "VANTOM_BOSS"}},
        });
        let mut probe = Offers(None);
        let _ = neow(&mut probe, &state, "v0.111.0", &opts);
        let (a, b, c) = probe.0.expect("the student was asked");
        assert_eq!(
            [norm_str(&a), norm_str(&b), norm_str(&c)],
            ["NEOWSTALISMAN", "PHIALHOLSTER", "LEAFYPOULTICE"]
        );
    }

    /// A Silent run's Neow is rebuilt as a Silent run, not an Ironclad one.
    #[test]
    fn a_neow_rebuild_plays_the_runs_character() {
        struct Who(Option<String>);
        impl Evaluator for Who {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = Some(format!("{:?}", rm.run_state.character_id));
                Err("probe".into())
            }
        }
        let options = [
            json!({"index": 0, "event_id": "NEOW", "relic_id": "NEOWS_TALISMAN"}),
            json!({"index": 1, "event_id": "NEOW", "relic_id": "PHIAL_HOLSTER"}),
            json!({"index": 2, "event_id": "NEOW", "relic_id": "LEAFY_POULTICE"}),
        ];
        let opts: Vec<&Value> = options.iter().collect();
        let state = json!({
            "seedless": true,
            "act1": "overgrowth",
            "run_state": {"ascension": 10, "character": "Silent"},
            "context": {"boss": {"id": "VANTOM_BOSS"}},
        });
        let mut probe = Who(None);
        let _ = neow(&mut probe, &state, "v0.111.0", &opts);
        assert_eq!(probe.0.as_deref(), Some("Silent"));
    }

    /// Stops at the page the screen showed and records its engine options.
    struct PageProbe(Vec<String>);
    impl Evaluator for PageProbe {
        fn evaluate(
            &mut self,
            rm: &mut RunManager,
            _: &[&str],
        ) -> std::result::Result<Pick, String> {
            self.0 = engine_event_tails(rm);
            Err("probe".into())
        }
    }

    /// An event page as a rebuild on `placeholder` rolls it (`None`: the
    /// default placeholder), as screen options.
    fn rolled_page(msg: &Value, event: &str, placeholder: Option<&str>) -> Vec<Value> {
        let mut msg = msg.clone();
        if let Some(p) = placeholder {
            msg["seedless_seed"] = json!(p);
        }
        let (mut rm, _) = graft_run(&msg, "v0.111.0").expect("graft");
        sts2_wire::graft::enter_event(&mut rm, event).expect("event");
        event_infos(&rm)
            .iter()
            .enumerate()
            .map(|(i, info)| {
                json!({"index": i, "label": "?", "event_id": event,
                       "text_key": format!("{event}.pages.INITIAL.options.{}", py::str_of(py::get(info, "option_id"))),
                       "vars": py::get(info, "vars").clone()})
            })
            .collect()
    }

    /// GH #1689: a later page is replayed on the engine's dice, which may
    /// upgrade another card than the game's Tablet of Truth did; the page
    /// holds the screen's deck.
    #[test]
    fn a_replayed_event_page_holds_the_screens_deck() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let name = "TABLET_OF_TRUTH";
        let mut first = seedless_message(&screen_map(&reference_run(1)), 1);
        first["run_state"]["master_deck"] =
            json!([{"id": "BASH", "upgraded": 0}, {"id": "ANGER", "upgraded": 0}]);
        first["type"] = json!("event");
        let key = |p: &str| format!("{name}.pages.{p}");
        first["options"] = json!([
            {"index": 0, "event_id": name, "text_key": key("INITIAL.options.DECIPHER_1")},
            {"index": 1, "event_id": name, "text_key": key("INITIAL.options.SMASH")},
        ]);
        let mut second = first.clone();
        second["options"] = json!([
            {"index": 0, "event_id": name, "text_key": key("DECIPHER_1.options.DECIPHER")},
            {"index": 1, "event_id": name, "text_key": key("DECIPHER.options.GIVE_UP")},
        ]);
        // The game upgraded ANGER, wherever the engine's dice land.
        second["run_state"]["master_deck"] =
            json!([{"id": "BASH", "upgraded": 0}, {"id": "ANGER", "upgraded": 1}]);
        let event_key = json!([Value::Null, first["floor"], "TABLETOFTRUTH"]);
        let track = json!({"key": event_key, "trail": [0], "pending": null, "origin": first});
        struct DeckProbe(Vec<(String, bool)>);
        impl Evaluator for DeckProbe {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = rm
                    .run_state
                    .master_deck
                    .iter()
                    .map(|c| (format!("{:?}", c.card_id), c.upgraded))
                    .collect();
                Err("probe".into())
            }
        }
        let mut probe = DeckProbe(Vec::new());
        let got = event(&mut probe, &second, "v0.111.0", Some(&track));
        assert_eq!(probe.0.len(), 2, "asked on the second page: {:?}", got.0);
        assert!(
            probe.0.iter().any(|(id, up)| id.contains("Anger") && *up)
                && probe.0.iter().any(|(id, up)| id.contains("Bash") && !*up),
            "{:?}",
            probe.0
        );
    }

    /// A lone-button page is a pick too: Tinker Time's first page has one
    /// button, so the trail replayed from it missed that pick, landed the
    /// third page's replay on the second (as many buttons as the screen) and
    /// read the card-type page's previews on the rider screen (Spire Bench
    /// seed 170000007).
    #[test]
    fn a_lone_button_page_counts_in_the_events_trail() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let name = "TINKER_TIME";
        let mut first = seedless_message(&screen_map(&reference_run(1)), 1);
        first["type"] = json!("event");
        let (mut rm, _) = graft_run(&first, "v0.111.0").expect("graft");
        sts2_wire::graft::enter_event(&mut rm, name).expect("event");
        let screen = |rm: &RunManager| -> Value {
            Value::Array(
                event_infos(rm)
                    .iter()
                    .enumerate()
                    .map(|(i, info)| {
                        json!({"index": i, "label": "?", "event_id": name,
                               "text_key": py::get(info, "loc_key").clone(),
                               "vars": py::get(info, "vars").clone()})
                    })
                    .collect(),
            )
        };
        first["options"] = screen(&rm);
        assert_eq!(first["options"].as_array().map(Vec::len), Some(1));
        assert!(choose_event(&mut rm, 0));
        let mut second = first.clone();
        second["options"] = screen(&rm);
        assert!(choose_event(&mut rm, 0));
        let mut third = first.clone();
        third["options"] = screen(&rm);
        let rider = engine_event_tails(&rm);
        let mut probe = PageProbe(Vec::new());
        let (_, track) = event(&mut probe, &first, "v0.111.0", None);
        let (_, track) = event(&mut probe, &second, "v0.111.0", track.as_ref());
        // The student answered the card-type page with its first button.
        let mut track = track.expect("a track");
        let shown: Vec<&Value> = second["options"]
            .as_array()
            .expect("options")
            .iter()
            .collect();
        track["pending"] = json!({"sig": page_sig(&second, &shown),
                                  "wire_to_engine": [[0, 0], [1, 1]], "best": 0, "human": null});
        let _ = event(&mut probe, &third, "v0.111.0", Some(&track));
        assert_eq!(probe.0, rider, "asked on the rider page");
    }

    /// A replayed later page holds the screen's belt: the potion an earlier
    /// page gave (Endless Conveyor's Suspicious Condiment) waits as a reward
    /// on the engine's replay but is in the game's belt (Spire Bench seed
    /// 170000008).
    #[test]
    fn a_replayed_event_page_holds_the_screens_potions() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let name = "TABLET_OF_TRUTH";
        let mut first = seedless_message(&screen_map(&reference_run(1)), 1);
        first["run_state"]["master_deck"] =
            json!([{"id": "BASH", "upgraded": 0}, {"id": "ANGER", "upgraded": 0}]);
        first["type"] = json!("event");
        let key = |p: &str| format!("{name}.pages.{p}");
        first["options"] = json!([
            {"index": 0, "event_id": name, "text_key": key("INITIAL.options.DECIPHER_1")},
            {"index": 1, "event_id": name, "text_key": key("INITIAL.options.SMASH")},
        ]);
        let mut second = first.clone();
        second["options"] = json!([
            {"index": 0, "event_id": name, "text_key": key("DECIPHER_1.options.DECIPHER")},
            {"index": 1, "event_id": name, "text_key": key("DECIPHER.options.GIVE_UP")},
        ]);
        second["run_state"]["potions"] = json!([
            {"slot": 0, "empty": false, "id": "Snecko Oil", "entry": "SNECKO_OIL"},
            {"slot": 1, "empty": false, "id": "FIRE_POTION"},
            {"slot": 2, "empty": true, "id": null},
        ]);
        let event_key = json!([Value::Null, first["floor"], "TABLETOFTRUTH"]);
        let track = json!({"key": event_key, "trail": [0], "pending": null, "origin": first});
        struct BeltProbe(Option<Vec<String>>);
        impl Evaluator for BeltProbe {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = Some(
                    rm.run_state
                        .potions
                        .iter()
                        .map(|p| p.python_name().to_owned())
                        .collect(),
                );
                Err("probe".into())
            }
        }
        let mut probe = BeltProbe(None);
        let got = event(&mut probe, &second, "v0.111.0", Some(&track));
        assert_eq!(
            probe.0,
            Some(vec!["SNECKO_OIL".to_owned(), "FIRE_POTION".to_owned()]),
            "{:?}",
            got.0
        );
    }

    /// GH #1691: The Future of Potions offers one button per potion under
    /// one key; the engine's options pair with them in order, not all with
    /// the last.
    #[test]
    fn buttons_that_share_a_key_pair_up_in_order() {
        let key = "THE_FUTURE_OF_POTIONS.pages.INITIAL.options.POTION";
        let screen: Vec<Value> = (0..3)
            .map(|i| json!({"index": i, "text_key": key}))
            .collect();
        let options: Vec<&Value> = screen.iter().collect();
        let tails = vec![norm_id(&json!(key)); 3];
        let got: Vec<i64> = match_options(&tails, &options)
            .iter()
            .map(|o| o.and_then(|o| o["index"].as_i64()).unwrap_or(-1))
            .collect();
        assert_eq!(got, vec![0, 1, 2]);
        // One key more than buttons still lands on a button.
        let four = vec![norm_id(&json!(key)); 4];
        assert!(match_options(&four, &options).iter().all(Option::is_some));
    }

    /// GH #1692: Tablet of Truth's DECIPHER_1..4 pages end in the same key
    /// segment; each one still records the pick that led to it.
    #[test]
    fn pages_that_share_short_keys_each_record_their_pick() {
        let page = |n: u32, max_hp: i64| {
            let state =
                json!({"run_state": {"hp": 60, "max_hp": max_hp, "gold": 99, "master_deck": []}});
            let opts = [
                json!({"text_key": format!("TABLET_OF_TRUTH.pages.DECIPHER_{n}.options.DECIPHER")}),
                json!({"text_key": "TABLET_OF_TRUTH.pages.DECIPHER.options.GIVE_UP"}),
            ];
            let sig = page_sig(&state, &opts.iter().collect::<Vec<_>>());
            (state, sig)
        };
        let mut track: Option<Value> = None;
        for (n, max_hp) in [(1, 77), (2, 71), (3, 62)] {
            let (state, sig) = page(n, max_hp);
            let mut t = event_track(track.as_ref(), &state, "TABLET_OF_TRUTH", &sig);
            t["pending"] = json!({"sig": sig, "best": 0, "human": null});
            track = Some(t);
        }
        let trail = py::list(track.as_ref().unwrap(), "trail").len();
        assert_eq!(trail, 2, "the picks on pages 1 and 2");
        // The same page asked again records nothing.
        let (state, sig) = page(3, 62);
        let again = event_track(track.as_ref(), &state, "TABLET_OF_TRUTH", &sig);
        assert_eq!(py::list(&again, "trail").len(), 2);
    }

    /// The game shows an unaffordable read as its own locked button; the
    /// engine leaves it out, so the page is matched without it.
    #[test]
    fn a_locked_button_is_left_out_of_the_page_match() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let mut state = seedless_message(&screen_map(&reference_run(1)), 1);
        let key = |t: &str| format!("SELF_HELP_BOOK.pages.INITIAL.options.{t}");
        state["options"] = json!([
            {"index": 0, "label": "?", "event_id": "SELF_HELP_BOOK", "text_key": key("READ_THE_BACK")},
            {"index": 1, "label": "?", "event_id": "SELF_HELP_BOOK", "text_key": key("READ_PASSAGE")},
            {"index": 2, "label": "?", "event_id": "SELF_HELP_BOOK", "text_key": key("READ_ENTIRE_BOOK_LOCKED"), "enabled": false},
        ]);
        let mut probe = PageProbe(Vec::new());
        let (out, _) = event(&mut probe, &state, "v0.111.0", None);
        assert!(
            matches!(out, Err(GraftError::Unavailable(ref m)) if m == "probe"),
            "{out:?}"
        );
    }

    /// Without the seed an event rolls its options on placeholder dice: the
    /// rebuild tries other placeholders until it shows the screen's page.
    #[test]
    fn a_seedless_event_rerolls_until_it_shows_the_screens_options() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let name = "COLORFUL_PHILOSOPHERS";
        let msg = seedless_message(&screen_map(&reference_run(1)), 1);
        let default = rolled_page(&msg, name, None);
        let screen = (1..200)
            .map(|k| rolled_page(&msg, name, Some(&format!("OTHER-{k}"))))
            .find(|p| {
                p.iter().map(|o| &o["text_key"]).collect::<Vec<_>>()
                    != default.iter().map(|o| &o["text_key"]).collect::<Vec<_>>()
            })
            .expect("some placeholder rolls other options");
        let mut state = msg.clone();
        state["type"] = json!("event");
        state["options"] = json!(screen);
        let mut probe = PageProbe(Vec::new());
        let (out, track) = event(&mut probe, &state, "v0.111.0", None);
        assert!(
            matches!(out, Err(GraftError::Unavailable(ref m)) if m == "probe"),
            "{out:?}"
        );
        let want: Vec<String> = screen.iter().map(|o| norm_id(&o["text_key"])).collect();
        assert_eq!(probe.0, want);
        let track = track.expect("track");
        assert!(
            track["origin"]["seedless_seed"].is_string(),
            "the later pages keep these dice"
        );
    }

    /// GH #1681: Tea Master's buttons name relics, but it is an event, not
    /// an Ancient's relic offer.
    #[test]
    fn an_event_whose_buttons_name_relics_stays_an_event() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let msg = seedless_message(&screen_map(&reference_run(1)), 1);
        // The game's buttons, as `_event_option` reads them: each tail a relic.
        let teas = ["BONE_TEA", "EMBER_TEA", "TEA_OF_DISCOURTESY"];
        let screen: Vec<Value> = teas
            .iter()
            .enumerate()
            .map(|(i, t)| {
                json!({"index": i, "label": "?", "event_id": "TEA_MASTER", "relic_id": t,
                       "text_key": format!("TEA_MASTER.pages.INITIAL.options.{t}")})
            })
            .collect();
        let mut state = msg.clone();
        state["type"] = json!("event");
        state["options"] = json!(screen);
        let mut probe = PageProbe(Vec::new());
        let _ = event(&mut probe, &state, "v0.111.0", None);
        let want: Vec<String> = teas.iter().map(|t| norm_str(t)).collect();
        assert_eq!(probe.0, want, "asked on the event's own page");
    }

    /// GH #1671: placeholder dice that roll the screen's buttons but not its
    /// amounts (WhisperingHollow's gold) are rerolled until they roll those
    /// too: the student reads them.
    #[test]
    fn a_seedless_event_rerolls_until_it_shows_the_screens_amounts() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let name = "WHISPERING_HOLLOW";
        let msg = seedless_message(&screen_map(&reference_run(1)), 1);
        let gold = |p: &[Value]| p.iter().find_map(|o| o["vars"]["Gold"].as_i64());
        let default = rolled_page(&msg, name, None);
        let screen = (1..400)
            .map(|k| rolled_page(&msg, name, Some(&format!("OTHER-{k}"))))
            .find(|p| gold(p).is_some() && gold(p) != gold(&default))
            .expect("some placeholder rolls another amount");
        let mut state = msg.clone();
        state["type"] = json!("event");
        state["options"] = json!(screen);
        struct GoldProbe(Option<i64>);
        impl Evaluator for GoldProbe {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = event_infos(rm)
                    .iter()
                    .find_map(|o| o["vars"]["Gold"].as_i64());
                Err("probe".into())
            }
        }
        let mut probe = GoldProbe(None);
        let _ = event(&mut probe, &state, "v0.111.0", None);
        assert_eq!(probe.0, gold(&screen), "the amount the screen shows");
    }

    /// GH #1646: an Ancient's relic that opens a deck screen at a later
    /// act's entrance (Biiig Hug's removal) leaves the run in no room, as
    /// the engine does, though the screen before it was the Ancient's event.
    #[test]
    fn a_deck_screen_at_a_later_acts_entrance_stands_in_no_room() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let reference = reference_run(2);
        let start = sts2_wire::emit::run_manager::full_map(&reference)["rows"][0][0].clone();
        let mut msg = seedless_message(&screen_map(&reference), 2);
        msg["map"]["current"] = json!({"col": start["col"], "row": 0});
        msg["map"]["visited"] = json!([]);
        msg["floor"] = json!(i64::from(reference.run_state.total_floor) + 2);
        let rm = graft_screen_run(&msg, "v0.111.0", Some("event")).expect("graft");
        assert_eq!(rm.run_state.act_floor, 0);
        assert!(
            rm.run_state.current_room.is_none(),
            "{:?}",
            rm.run_state.current_room
        );
    }

    /// GH #1686: Jungle Maze rolls two golds (~900 pairs); a seedless
    /// rebuild finds a placeholder that rolls the screen's exact pair.
    #[test]
    fn a_seedless_jungle_maze_rolls_the_screens_two_golds() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let name = "JUNGLE_MAZE_ADVENTURE";
        let msg = seedless_message(&screen_map(&reference_run(1)), 1);
        let var = |p: &[Value], k: &str| p.iter().find_map(|o| o["vars"][k].as_i64());
        let mut screen = rolled_page(&msg, name, None);
        let (solo, join) = (var(&screen, "SoloGold"), var(&screen, "JoinForcesGold"));
        assert!(
            solo.is_some() && join.is_some(),
            "the engine names its golds"
        );
        // A pair the placeholder dice did not roll.
        let (want_solo, want_join) = (136, 61);
        for (o, key) in screen.iter_mut().zip(["SOLO_QUEST", "JOIN_FORCES"]) {
            o["text_key"] = json!(format!("{name}.pages.INITIAL.options.{key}"));
            for (k, v) in [("SoloGold", want_solo), ("JoinForcesGold", want_join)] {
                if o["vars"].get(k).is_some() {
                    o["vars"][k] = json!(v);
                }
            }
        }
        let mut state = msg.clone();
        state["type"] = json!("event");
        state["options"] = json!(screen);
        struct GoldProbe(Option<(i64, i64)>);
        impl Evaluator for GoldProbe {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                let infos = event_infos(rm);
                let var = |k: &str| infos.iter().find_map(|o| o["vars"][k].as_i64());
                self.0 = var("SoloGold").zip(var("JoinForcesGold"));
                Err("probe".into())
            }
        }
        let mut probe = GoldProbe(None);
        let got = event(&mut probe, &state, "v0.111.0", None);
        let (s, j) = probe
            .0
            .unwrap_or_else(|| panic!("asked on the event's page: {got:?}"));
        assert_ne!(
            (solo, join),
            (Some(want_solo), Some(want_join)),
            "the default dice differ"
        );
        assert_eq!((s, j), (want_solo, want_join), "the screen's golds");
    }

    /// Sunken Treasury rolls two golds (16 x 61 pairs); a seedless rebuild
    /// finds a placeholder that rolls the screen's exact pair (Spire Bench
    /// seeds 170000020 and 170000035 read the small chest 1 gold off).
    #[test]
    fn a_seedless_sunken_treasury_rolls_the_screens_two_golds() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let name = "SUNKEN_TREASURY";
        let msg = seedless_message(&screen_map(&reference_run(1)), 1);
        let var = |p: &[Value], k: &str| p.iter().find_map(|o| o["vars"][k].as_i64());
        let mut screen = rolled_page(&msg, name, None);
        let (small, large) = (
            var(&screen, "SmallChestGold"),
            var(&screen, "LargeChestGold"),
        );
        assert!(
            small.is_some() && large.is_some(),
            "the engine names its golds"
        );
        for (want_small, want_large) in [(53, 361), (67, 304), (60, 333)] {
            for (o, key) in screen.iter_mut().zip(["FIRST_CHEST", "SECOND_CHEST"]) {
                o["text_key"] = json!(format!("{name}.pages.INITIAL.options.{key}"));
                for (k, v) in [
                    ("SmallChestGold", want_small),
                    ("LargeChestGold", want_large),
                ] {
                    if o["vars"].get(k).is_some() {
                        o["vars"][k] = json!(v);
                    }
                }
            }
            let mut state = msg.clone();
            state["type"] = json!("event");
            state["options"] = json!(screen);
            struct GoldProbe(Option<(i64, i64)>);
            impl Evaluator for GoldProbe {
                fn evaluate(
                    &mut self,
                    rm: &mut RunManager,
                    _: &[&str],
                ) -> std::result::Result<Pick, String> {
                    let infos = event_infos(rm);
                    let var = |k: &str| infos.iter().find_map(|o| o["vars"][k].as_i64());
                    self.0 = var("SmallChestGold").zip(var("LargeChestGold"));
                    Err("probe".into())
                }
            }
            let mut probe = GoldProbe(None);
            let got = event(&mut probe, &state, "v0.111.0", None);
            let got = probe
                .0
                .unwrap_or_else(|| panic!("asked on the event's page: {got:?}"));
            assert_eq!(got, (want_small, want_large), "the screen's golds");
        }
    }

    /// GH #1680: Wood Carvings lists only the deck's Basic cards; the
    /// engine's transform then offers those, not every transformable one.
    #[test]
    fn a_basic_only_transform_screen_offers_only_the_basic_cards() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let mut state = seedless_message(&screen_map(&reference_run(1)), 1);
        state["run_state"]["master_deck"] =
            json!([{"id": "BASH", "upgraded": 0}, {"id": "ANGER", "upgraded": 0}]);
        state["type"] = json!("card_select");
        state["prompt_key"] = json!("TO_TRANSFORM");
        state["cards"] = json!([{"index": 0, "id": "BASH"}]);
        state["min_count"] = json!(1);
        state["max_count"] = json!(1);
        struct Offer(Option<usize>);
        impl Evaluator for Offer {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = Some(rm.transform_eligible_deck_indices().len());
                Err("probe".into())
            }
        }
        let mut probe = Offer(None);
        let _ = deck_select(&mut probe, &state, "v0.111.0", None, None);
        assert_eq!(probe.0, Some(1), "Bash only");
        // A screen listing the whole deck is a plain transform.
        state["cards"] = json!([{"index": 0, "id": "BASH"}, {"index": 1, "id": "ANGER"}]);
        let _ = deck_select(&mut probe, &state, "v0.111.0", None, None);
        assert_eq!(probe.0, Some(2), "the whole deck");
    }

    /// Pael's Tooth lists only the deck's upgradable cards
    /// (`PaelsTooth.AfterObtained`, filter `c.IsUpgradable`); the engine's
    /// removal then offers those, not every removable one (Spire Bench seed
    /// 180000416).
    #[test]
    fn an_upgradable_only_removal_screen_offers_only_the_upgradable_cards() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let mut state = seedless_message(&screen_map(&reference_run(1)), 1);
        state["run_state"]["master_deck"] = json!([
            {"id": "STRIKE_IRONCLAD", "upgraded": 1},
            {"id": "STRIKE_IRONCLAD", "upgraded": 0},
            {"id": "BASH", "upgraded": 0}
        ]);
        state["type"] = json!("card_select");
        state["prompt_key"] = json!("TO_REMOVE");
        state["cards"] = json!([
            {"index": 0, "id": "STRIKE_IRONCLAD"},
            {"index": 1, "id": "BASH"}
        ]);
        state["min_count"] = json!(1);
        state["max_count"] = json!(1);
        struct Offer(Option<usize>);
        impl Evaluator for Offer {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = Some(rm.run_state.removal_offer_order().len());
                Err("probe".into())
            }
        }
        let mut probe = Offer(None);
        let _ = deck_select(&mut probe, &state, "v0.111.0", None, None);
        assert_eq!(probe.0, Some(2), "the upgradable Strike and Bash");
        // A screen listing every removable card is a plain removal.
        state["cards"] = json!([
            {"index": 0, "id": "STRIKE_IRONCLAD", "upgraded": true},
            {"index": 1, "id": "STRIKE_IRONCLAD"},
            {"index": 2, "id": "BASH"}
        ]);
        let _ = deck_select(&mut probe, &state, "v0.111.0", None, None);
        assert_eq!(probe.0, Some(3), "the whole deck");
    }

    /// GH #1732: a card reward at a later act's entrance (Glass Eye's five
    /// at Orobas) stands in no room, as the engine does, though the mod
    /// names the Ancient's event room (Spire Bench seed 180001539).
    #[test]
    fn a_card_reward_at_a_later_acts_entrance_stands_in_no_room() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let reference = reference_run(2);
        let start = sts2_wire::emit::run_manager::full_map(&reference)["rows"][0][0].clone();
        let mut msg = seedless_message(&screen_map(&reference), 2);
        msg["map"]["current"] = json!({"col": start["col"], "row": 0});
        msg["map"]["visited"] = json!([]);
        msg["floor"] = json!(i64::from(reference.run_state.total_floor) + 2);
        msg["current_room"] = json!("Event");
        msg["type"] = json!("card_reward");
        msg["cards"] = json!([
            {"index": 0, "id": "POMMEL_STRIKE"},
            {"index": 1, "id": "ANGER"},
            {"index": 2, "id": "TREMBLE"}
        ]);
        struct Room(Option<Option<sts2_core::runstate::RoomType>>);
        impl Evaluator for Room {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = Some(rm.run_state.current_room.map(|r| r.room_type));
                Err("probe".into())
            }
        }
        let mut probe = Room(None);
        let got = card_reward(&mut probe, &msg, "v0.111.0", Some("Event"));
        assert_eq!(probe.0, Some(None), "asked in no room: {got:?}");
    }

    /// GH #1763: a deck screen's card takes its deck card's enchantment, in
    /// deck order.
    #[test]
    fn a_deck_screens_cards_take_the_decks_enchantments() {
        let state = json!({
            "cards": [
                {"index": 0, "id": "CARD.POMMEL_STRIKE", "upgraded": false},
                {"index": 1, "id": "CARD.POMMEL_STRIKE", "upgraded": false}
            ],
            "run_state": {"master_deck": [
                {"id": "CARD.POMMEL_STRIKE", "upgraded": true, "enchantment": "Adroit"},
                {"id": "CARD.POMMEL_STRIKE", "upgraded": false, "enchantment": "Adroit"},
                {"id": "CARD.BASH", "upgraded": false},
                {"id": "CARD.POMMEL_STRIKE", "upgraded": false}
            ]}
        });
        let keys: Vec<CardKey> = with_deck_enchantments(&state)
            .iter()
            .map(wire_card_key)
            .collect();
        assert_eq!(keys[0].2, norm_id(&json!("Adroit")));
        assert_eq!(keys[1].2, "");
        let plain = (keys[1].0.clone(), false, String::new());
        let cards = with_deck_enchantments(&state);
        let refs: Vec<&Value> = cards.iter().collect();
        assert_eq!(find_card(&refs, &plain), Some(1), "the plain copy");
    }

    /// GH #1767: a removal lists the curses first; each Strike still takes
    /// its own deck copy's enchantment.
    #[test]
    fn a_reordered_deck_screen_keeps_each_copys_enchantment() {
        let state = json!({
            "cards": [
                {"index": 0, "id": "CARD.CLUMSY", "upgraded": false},
                {"index": 1, "id": "CARD.STRIKE_IRONCLAD", "upgraded": false},
                {"index": 2, "id": "CARD.STRIKE_IRONCLAD", "upgraded": false}
            ],
            "run_state": {"master_deck": [
                {"id": "CARD.STRIKE_IRONCLAD", "upgraded": false, "enchantment": "Spiral"},
                {"id": "CARD.STRIKE_IRONCLAD", "upgraded": false},
                {"id": "CARD.ASCENDERS_BANE", "upgraded": false},
                {"id": "CARD.CLUMSY", "upgraded": false}
            ]}
        });
        let keys: Vec<CardKey> = with_deck_enchantments(&state)
            .iter()
            .map(wire_card_key)
            .collect();
        assert_eq!(keys[1].2, norm_id(&json!("Spiral")));
        assert_eq!(keys[2].2, "");
    }

    /// Corpus seed 180000183: Spiraling Whirlpool's screen leaves out the
    /// Nimble Defend (an enchanted card cannot take a second enchantment), so
    /// its first Defend is the deck's SECOND one. The engine's pick (the first
    /// plain Defend) must land on screen index 5, not 6.
    #[test]
    fn an_enchant_screens_cards_are_all_unenchanted() {
        let strike = || json!({"id": "CARD.STRIKE_IRONCLAD", "upgraded": false});
        let defend = || json!({"id": "CARD.DEFEND_IRONCLAD", "upgraded": false});
        let mut deck: Vec<Value> = (0..5).map(|_| strike()).collect();
        deck.push(json!({"id": "CARD.DEFEND_IRONCLAD", "upgraded": false, "enchantment": "Nimble", "enchantment_amount": 2}));
        deck.extend((0..3).map(|_| defend()));
        let mut cards: Vec<Value> = (0..5).map(|_| strike()).collect();
        cards.extend((0..3).map(|_| defend()));
        for (i, c) in cards.iter_mut().enumerate() {
            c["index"] = json!(i);
        }
        let state = json!({
            "prompt_key": "TO_ENCHANT",
            "source_model": "SPIRAL",
            "cards": cards,
            "run_state": {"master_deck": deck},
        });
        let enriched = with_deck_enchantments(&state);
        assert!(
            enriched.iter().all(|c| c.get("enchantment").is_none()),
            "no enchant-screen card is enchanted: {enriched:?}"
        );
        let refs: Vec<&Value> = enriched.iter().collect();
        let plain_defend = wire_card_key(&defend());
        assert_eq!(
            find_card(&refs, &plain_defend),
            Some(5),
            "the first Defend on the screen"
        );
    }

    /// GH #1750: Dolly's Mirror's copy is the engine's card grant.
    #[test]
    fn dollys_mirror_is_a_card_grant() {
        assert_eq!(
            prompt_kind(&"DOLLYS_MIRROR.selectionScreenPrompt".to_uppercase()),
            Some("grant")
        );
    }

    /// GH #1747: Amalgamator's removal offers its Basic Defends only, as
    /// the engine does, not the whole deck.
    #[test]
    fn amalgamator_offers_its_basic_cards_only() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let mut state = seedless_message(&screen_map(&reference_run(1)), 1);
        state["run_state"]["master_deck"] = json!([
            {"id": "STRIKE_IRONCLAD", "upgraded": 0},
            {"id": "DEFEND_IRONCLAD", "upgraded": 0},
            {"id": "DEFEND_IRONCLAD", "upgraded": 1},
            {"id": "DEFEND_IRONCLAD", "upgraded": 0},
            {"id": "BASH", "upgraded": 0}
        ]);
        state["type"] = json!("card_select");
        state["prompt_key"] = json!("TO_REMOVE");
        state["event_id"] = json!("AMALGAMATOR");
        state["cards"] = json!([
            {"index": 1, "id": "DEFEND_IRONCLAD"},
            {"index": 2, "id": "DEFEND_IRONCLAD", "upgraded": true},
            {"index": 3, "id": "DEFEND_IRONCLAD"}
        ]);
        state["min_count"] = json!(2);
        state["max_count"] = json!(2);
        struct Offer(Option<usize>);
        impl Evaluator for Offer {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = Some(rm.run_state.removal_offer_order().len());
                Err("probe".into())
            }
        }
        let mut probe = Offer(None);
        let _ = deck_select(&mut probe, &state, "v0.111.0", Some("event"), None);
        assert_eq!(probe.0, Some(3), "the three Defends");
        // The second pick replays the first in the engine.
        for first in 0..3 {
            state["picked_indices"] = json!([first]);
            probe.0 = None;
            let second = deck_select(&mut probe, &state, "v0.111.0", Some("event"), None);
            assert_eq!(probe.0, Some(2), "first pick {first}: {second:?}");
        }
    }

    /// GH #1715: Kifuda's pick-3 enchant asked one pick at a time holds the
    /// earlier picks in the engine, so an enchanted card leaves the offer as
    /// it does natively.
    #[test]
    fn a_pick_three_enchant_replays_the_first_pick_in_the_engine() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let mut state = seedless_message(&screen_map(&reference_run(1)), 1);
        state["run_state"]["master_deck"] = json!([
            {"id": "BASH", "upgraded": 0},
            {"id": "ANGER", "upgraded": 0},
            {"id": "TAUNT", "upgraded": 0}
        ]);
        state["type"] = json!("card_select");
        state["prompt_key"] = json!("TO_ENCHANT");
        state["pending_enchantment"] = json!("ADROIT");
        state["enchantment_amount"] = json!(3);
        state["cards"] = json!([
            {"index": 0, "id": "BASH"},
            {"index": 1, "id": "ANGER"},
            {"index": 2, "id": "TAUNT"}
        ]);
        state["min_count"] = json!(0);
        state["max_count"] = json!(3);
        state["picked_indices"] = json!([2]);
        struct Offer(Option<usize>);
        impl Evaluator for Offer {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = Some(rm.enchant_eligible_deck_indices().len());
                Err("probe".into())
            }
        }
        let mut probe = Offer(None);
        let got = deck_select(&mut probe, &state, "v0.111.0", None, None);
        assert_eq!(
            probe.0,
            Some(2),
            "Taunt is enchanted and off the offer: {got:?}"
        );
    }

    /// The enchant offer the rebuilt run makes for `cards`, on `deck`.
    fn rebuilt_enchant_offer(deck: Value, enchantment: &str, cards: Value) -> Option<usize> {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let mut state = seedless_message(&screen_map(&reference_run(1)), 1);
        state["run_state"]["master_deck"] = deck;
        state["type"] = json!("card_select");
        state["prompt_key"] = json!("TO_ENCHANT");
        state["pending_enchantment"] = json!(enchantment);
        state["enchantment_amount"] = json!(1);
        state["cards"] = cards;
        state["min_count"] = json!(1);
        state["max_count"] = json!(1);
        struct Offer(Option<usize>);
        impl Evaluator for Offer {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = Some(rm.enchant_eligible_deck_indices().len());
                Err("probe".into())
            }
        }
        let mut probe = Offer(None);
        let _ = deck_select(&mut probe, &state, "v0.111.0", None, None);
        probe.0
    }

    /// Royal Stamp's Royally Approved enchants an Attack or a Skill: the
    /// rebuild offered the deck's Powers too (seeds 180001793, 180000174).
    #[test]
    fn a_royally_approved_screen_offers_no_power() {
        let deck = json!([
            {"id": "BASH", "upgraded": 0},
            {"id": "TAUNT", "upgraded": 0},
            {"id": "INFLAME", "upgraded": 0}
        ]);
        let cards = json!([
            {"index": 0, "id": "BASH", "type": "Attack"},
            {"index": 1, "id": "TAUNT", "type": "Skill"}
        ]);
        assert_eq!(
            rebuilt_enchant_offer(deck, "ROYALLY_APPROVED", cards),
            Some(2)
        );
    }

    /// Mad Science keeps the type Tinker Time gave it: a Power one is on
    /// Swift's offer (seeds 180001862, 180001815, 180001497) and off
    /// Momentum's (seeds 180000660, 180000856).
    #[test]
    fn a_power_mad_science_is_offered_as_a_power() {
        let deck = json!([
            {"id": "BASH", "upgraded": 0},
            {"id": "INFLAME", "upgraded": 0},
            {"id": "MAD_SCIENCE", "upgraded": 0, "type": "Power", "tinker_time_rider": "Curious"}
        ]);
        let powers = json!([
            {"index": 0, "id": "INFLAME", "type": "Power"},
            {"index": 1, "id": "MAD_SCIENCE", "type": "Power"}
        ]);
        assert_eq!(
            rebuilt_enchant_offer(deck.clone(), "SWIFT", powers),
            Some(2)
        );
        let attacks = json!([{"index": 0, "id": "BASH", "type": "Attack"}]);
        assert_eq!(rebuilt_enchant_offer(deck, "MOMENTUM", attacks), Some(1));
    }

    /// GH #1673: a pick-N grant asked one pick at a time holds the earlier
    /// picks in the engine, so its minimum falls as the screen fills.
    #[test]
    fn a_pick_two_grant_replays_the_first_pick_in_the_engine() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let mut state = seedless_message(&screen_map(&reference_run(1)), 1);
        state["type"] = json!("card_select");
        state["prompt_key"] = json!("ROOM_FULL_OF_CHEESE.pages.GORGE.selectionScreenPrompt");
        let ids = ["POMMEL_STRIKE", "SHRUG_IT_OFF", "ANGER"];
        state["cards"] = json!(ids
            .iter()
            .enumerate()
            .map(|(i, c)| json!({"index": i, "id": c}))
            .collect::<Vec<_>>());
        state["min_count"] = json!(2);
        state["max_count"] = json!(2);
        state["picked_indices"] = json!([1]);
        struct GrantProbe(Option<(usize, usize, usize)>, Option<i64>);
        impl Evaluator for GrantProbe {
            fn set_grant_picked(&mut self, n: i64) {
                self.1 = Some(n);
            }
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = rm
                    .run_state
                    .pending_card_grants
                    .first()
                    .map(|g| (g.pick, g.min_pick, g.options.len()));
                Err("probe".into())
            }
        }
        let mut probe = GrantProbe(None, None);
        let _ = deck_select(&mut probe, &state, "v0.111.0", None, None);
        assert_eq!(
            probe.0,
            Some((1, 1, 2)),
            "one pick left of the two cards left"
        );
        // GH #1677: and the student is told one pick is taken.
        assert_eq!(probe.1, Some(1), "picks taken");
    }

    /// GH #1736: Brain Leech's Share Knowledge grid is the engine's
    /// choose-a-card screen with no skip, as the native run offers it.
    #[test]
    fn brain_leechs_share_knowledge_is_a_choose_screen_without_skip() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let mut state = seedless_message(&screen_map(&reference_run(1)), 1);
        state["type"] = json!("card_select");
        state["prompt_key"] = json!("BRAIN_LEECH.pages.SHARE_KNOWLEDGE.selectionScreenPrompt");
        let ids = [
            "BLUDGEON",
            "RUPTURE_CARD",
            "TAUNT",
            "PILLAGE",
            "TWIN_STRIKE",
        ];
        state["cards"] = json!(ids
            .iter()
            .enumerate()
            .map(|(i, c)| json!({"index": i, "id": c}))
            .collect::<Vec<_>>());
        state["min_count"] = json!(1);
        state["max_count"] = json!(1);
        // (pick buttons, a skip button, queued grants) the student sees.
        struct OfferProbe(Option<(usize, bool, usize)>);
        impl Evaluator for OfferProbe {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                use sts2_core::RunAction;
                let actions = rm.get_available_actions();
                let picks = actions
                    .iter()
                    .filter(|a| matches!(a, RunAction::PickCardReward { idx: Some(_) }))
                    .count();
                let skip = actions.contains(&RunAction::PickCardReward { idx: None });
                self.0 = Some((picks, skip, rm.run_state.pending_card_grants.len()));
                Err("probe".into())
            }
        }
        let mut probe = OfferProbe(None);
        let _ = deck_select(&mut probe, &state, "v0.111.0", Some("event"), None);
        assert_eq!(probe.0, Some((5, false, 0)));
    }

    /// GH #1739: Trial's three trials all offer Guilty / Innocent. A
    /// seedless rebuild rerolls until it shows the trial the screen names.
    #[test]
    fn a_seedless_trial_page_is_the_screens_trial() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let msg = seedless_message(&screen_map(&reference_run(2)), 2);
        struct PageProbe(Option<String>);
        impl Evaluator for PageProbe {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = event_infos(rm)
                    .first()
                    .map(|i| page_of(py::get(i, "loc_key")));
                Err("probe".into())
            }
        }
        for trial in ["MERCHANT", "NOBLE", "NONDESCRIPT"] {
            let mut state = msg.clone();
            state["type"] = json!("event");
            state["options"] = json!(["GUILTY", "INNOCENT"]
                .iter()
                .enumerate()
                .map(|(i, o)| json!({"index": i, "label": o, "event_id": "TRIAL",
                                     "text_key": format!("TRIAL.pages.{trial}.options.{o}")}))
                .collect::<Vec<_>>());
            let mut probe = PageProbe(None);
            let _ = event(&mut probe, &state, "v0.111.0", None);
            assert_eq!(probe.0, Some(format!("TRIAL.PAGES.{trial}")), "{trial}");
        }
    }

    /// A seedless rebuild of Doll Room's second page rerolls until the
    /// engine offers the dolls the screen shows, in its order. The engine
    /// names a doll button only by its label, so the check compares labels
    /// with the screen's titles.
    #[test]
    fn a_seedless_doll_room_offers_the_screens_dolls() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let msg = seedless_message(&screen_map(&reference_run(2)), 2);
        struct PageProbe(Vec<String>);
        impl Evaluator for PageProbe {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = event_infos(rm)
                    .iter()
                    .map(|i| py::str_of(py::get(i, "label")))
                    .collect();
                Err("probe".into())
            }
        }
        let mut origin = msg.clone();
        origin["type"] = json!("event");
        let titles = [
            ("Daughter of the Wind", "DaughterOfTheWind"),
            ("Mr. Struggles", "MrStruggles"),
            ("Bing Bong", "BingBong"),
        ];
        for (a, b) in [(0, 2), (2, 0), (1, 0), (0, 1), (2, 1), (1, 2)] {
            let mut state = origin.clone();
            state["options"] = json!([titles[a].0, titles[b].0]
                .iter()
                .enumerate()
                .map(|(i, t)| json!({"index": i, "label": t, "title": t,
                                     "event_id": "DOLL_ROOM", "text_key": t}))
                .collect::<Vec<_>>());
            let track = json!({
                "key": [py::get(&state, "seed"), py::get(&state, "floor"), "DOLLROOM"],
                "trail": [1], "pending": null, "origin": origin,
            });
            let mut probe = PageProbe(Vec::new());
            let _ = event(&mut probe, &state, "v0.111.0", Some(&track));
            assert_eq!(probe.0, vec![titles[a].1, titles[b].1], "{a} {b}");
        }
    }

    /// Endless Conveyor's dish after a grab rolls on the event's dice. A
    /// seedless rebuild rerolls until the belt holds the screen's dish: its
    /// button shares the page and the Leave button with any other dish's
    /// (Spire Bench seed 170000008).
    #[test]
    fn a_seedless_conveyor_belt_holds_the_screens_dish() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        let msg = seedless_message(&screen_map(&reference_run(2)), 2);
        struct DishProbe(Option<String>);
        impl Evaluator for DishProbe {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = event_infos(rm)
                    .first()
                    .map(|i| py::str_of(py::get(i, "loc_key")));
                Err("probe".into())
            }
        }
        let mut origin = msg.clone();
        origin["type"] = json!("event");
        let leave = "ENDLESS_CONVEYOR.pages.GRAB_SOMETHING_OFF_THE_BELT.options.LEAVE";
        for dish in ["CAVIAR", "CLAM_ROLL", "JELLY_LIVER", "FRIED_EEL"] {
            let key = format!("ENDLESS_CONVEYOR.pages.ALL.options.{dish}");
            let mut state = origin.clone();
            state["options"] = json!([
                {"index": 0, "label": dish, "event_id": "ENDLESS_CONVEYOR", "text_key": key},
                {"index": 1, "label": "Leave", "event_id": "ENDLESS_CONVEYOR", "text_key": leave},
            ]);
            let track = json!({
                "key": [py::get(&state, "seed"), py::get(&state, "floor"), "ENDLESSCONVEYOR"],
                "trail": [0], "pending": null, "origin": origin,
            });
            let mut probe = DishProbe(None);
            let _ = event(&mut probe, &state, "v0.111.0", Some(&track));
            assert_eq!(probe.0.as_deref(), Some(key.as_str()), "{dish}");
        }
    }

    /// A grab whose dish queues a deck choice (Jelly Liver's transform) is
    /// resolved on the game's own card screen before the belt's next page
    /// shows. The page replayed from the event's first one must not keep
    /// that transform queued: the rebuilt run would stand at
    /// `CARD_TRANSFORM`, not the page (Spire Bench seeds 180001043,
    /// 180000804 and 15 more, corpus r5).
    #[test]
    fn a_conveyor_page_after_jelly_liver_has_no_pending_transform() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        sts2_e2e::env::ensure_engine_installed();
        let mut msg = seedless_message(&screen_map(&reference_run(2)), 2);
        // Two cards: with one, the transform's select takes it on its own
        // and nothing stays queued.
        msg["run_state"]["master_deck"] =
            json!([{"id": "BASH", "upgraded": 0}, {"id": "STRIKE_IRONCLAD", "upgraded": 0}]);
        let jelly = "ENDLESS_CONVEYOR.pages.ALL.options.JELLY_LIVER";
        let leave = "ENDLESS_CONVEYOR.pages.GRAB_SOMETHING_OFF_THE_BELT.options.LEAVE";
        // A placeholder whose first page offers Jelly Liver, and the dish
        // the belt holds after it.
        let (origin, next) = (1..=512)
            .find_map(|n| {
                let mut origin = msg.clone();
                origin["type"] = json!("event");
                origin["seedless_seed"] = json!(format!("{}-j{n}", crate::graft::SEEDLESS_SEED));
                let (mut rm, _) = graft_run(&origin, "v0.111.0").ok()?;
                sts2_wire::graft::enter_event(&mut rm, "ENDLESS_CONVEYOR").ok()?;
                let first = event_infos(&rm)
                    .first()
                    .map(|i| py::str_of(py::get(i, "loc_key")))?;
                if first != jelly {
                    return None;
                }
                assert!(choose_event(&mut rm, 0) && in_event(&rm));
                assert!(
                    !rm.run_state.pending_deck_transforms.is_empty()
                        || rm.run_state.has_pending_grant_choice(),
                    "the engine queues Jelly Liver's transform"
                );
                let next = event_infos(&rm)
                    .first()
                    .map(|i| py::str_of(py::get(i, "loc_key")))?;
                Some((origin, next))
            })
            .expect("a placeholder rolls Jelly Liver first");
        struct PendingProbe(Option<(String, bool)>);
        impl Evaluator for PendingProbe {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                let key = event_infos(rm)
                    .first()
                    .map(|i| py::str_of(py::get(i, "loc_key")))
                    .unwrap_or_default();
                self.0 = Some((key, rm.run_state.has_pending_grant_choice()));
                Err("probe".into())
            }
        }
        let mut state = origin.clone();
        state["options"] = json!([
            {"index": 0, "label": "dish", "event_id": "ENDLESS_CONVEYOR", "text_key": next},
            {"index": 1, "label": "Leave", "event_id": "ENDLESS_CONVEYOR", "text_key": leave},
        ]);
        let track = json!({
            "key": [py::get(&state, "seed"), py::get(&state, "floor"), "ENDLESSCONVEYOR"],
            "trail": [0], "pending": null, "origin": origin,
        });
        let mut probe = PendingProbe(None);
        let _ = event(&mut probe, &state, "v0.111.0", Some(&track));
        let (key, pending) = probe.0.expect("the page is evaluated");
        assert_eq!(key, next, "the belt's next page");
        assert!(!pending, "the game resolved the transform before this page");
    }

    /// The belt's dish is the screen's even when no placeholder replay can
    /// roll it: the run stands at full HP, so the engine's pool holds no
    /// Clam Roll after the first grab, while the game's player (who lost
    /// HP since) was offered one (seeds 180000706, 180000544, 180001732
    /// rolled other dishes on long visits).
    #[test]
    fn a_conveyor_page_shows_the_screens_dish() {
        use crate::graft::tests::{reference_run, screen_map, seedless_message};
        sts2_e2e::env::ensure_engine_installed();
        let mut origin = seedless_message(&screen_map(&reference_run(2)), 2);
        origin["type"] = json!("event");
        origin["run_state"]["hp"] = json!(80);
        let clam = "ENDLESS_CONVEYOR.pages.ALL.options.CLAM_ROLL";
        let leave = "ENDLESS_CONVEYOR.pages.GRAB_SOMETHING_OFF_THE_BELT.options.LEAVE";
        struct DishProbe(Option<String>);
        impl Evaluator for DishProbe {
            fn evaluate(
                &mut self,
                rm: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                self.0 = event_infos(rm)
                    .first()
                    .map(|i| py::str_of(py::get(i, "loc_key")));
                Err("probe".into())
            }
        }
        let mut state = origin.clone();
        state["run_state"]["hp"] = json!(70);
        state["options"] = json!([
            {"index": 0, "label": "dish", "event_id": "ENDLESS_CONVEYOR", "text_key": clam},
            {"index": 1, "label": "Leave", "event_id": "ENDLESS_CONVEYOR", "text_key": leave},
        ]);
        let track = json!({
            "key": [py::get(&state, "seed"), py::get(&state, "floor"), "ENDLESSCONVEYOR"],
            "trail": [0], "pending": null, "origin": origin,
        });
        let mut probe = DishProbe(None);
        let _ = event(&mut probe, &state, "v0.111.0", Some(&track));
        assert_eq!(probe.0.as_deref(), Some(clam));
    }

    /// GH #1605: on Trial's REJECT page the engine (agent mode) offers only
    /// Accept, so Instinct answers with the one surviving button instead of
    /// failing to match the screen.
    #[test]
    fn trial_reject_page_answers_accept() {
        let state = json!({"options": [
            {"index": 0, "label": "Accept", "event_id": "TRIAL",
             "text_key": "TRIAL.pages.REJECT.options.ACCEPT", "will_kill_player": false},
            {"index": 1, "label": "Double Down", "event_id": "TRIAL",
             "text_key": "TRIAL.pages.REJECT.options.DOUBLE_DOWN", "will_kill_player": true},
        ]});
        struct Unused;
        impl Evaluator for Unused {
            fn evaluate(
                &mut self,
                _: &mut RunManager,
                _: &[&str],
            ) -> std::result::Result<Pick, String> {
                panic!("a lone option needs no net")
            }
        }
        let mut ev = Unused;
        let (advice, _) = event(&mut ev, &state, "v0.111.0", None);
        let advice = advice.expect("one option left");
        assert_eq!(advice["best"]["index"], json!(0));
    }

    /// GH #1607: the mod's Fake Merchant shop grafts as the engine's
    /// event-room shop (rolled prices, no removal, leaving ends the event).
    #[test]
    fn the_fake_merchant_shop_grafts_as_its_event_shop() {
        let buy = |i: i64, id: &str, price: i64| json!({"index": i, "action": "buy_relic", "price": price, "item": {"id": id}});
        let (a, b) = (buy(1, "FAKE_MANGO", 53), buy(2, "FAKE_ANCHOR", 44));
        let state = json!({"event_name": "FAKE_MERCHANT"});
        let inv = shop_inventory(&state, &[], &[&a, &b], &[], None, &Value::Null);
        assert_eq!(inv["fake_merchant"], json!(true));
        let regular = shop_inventory(&json!({}), &[], &[&a], &[], None, &Value::Null);
        assert_eq!(regular["fake_merchant"], json!(false));

        sts2_content::relics::dispatch::ensure_installed();
        let mut rm = sts2_content::run_callbacks::new_run_manager_for_character(
            7,
            sts2_core::CharacterId::Ironclad,
        );
        rm.run_state.relics.push(sts2_core::RelicId::MembershipCard);
        rm.run_state.gold = 100;
        sts2_wire::graft::open_shop(&mut rm, &inv).expect("opens");
        assert!(rm.in_fake_merchant_shop());
        let _ = rm.take_action(sts2_core::run_manager::RunAction::ShopBuyRelic { idx: 0 });
        assert_eq!(rm.run_state.gold, 47, "the rolled price, undiscounted");
    }

    #[test]
    fn ids_normalise_across_spellings() {
        assert_eq!(norm_id(&json!("CARD.STRIKE_IRONCLAD")), "STRIKEIRONCLAD");
        assert_eq!(norm_id(&json!("StrikeIronclad")), "STRIKEIRONCLAD");
        assert_eq!(norm_id(&Value::Null), "");
    }

    #[test]
    fn the_swap_group_is_the_first_button_decline_first() {
        let opts = vec![
            json!({"action": "potion_swap", "reward_button": 2, "discard_slot": 0}),
            json!({"action": "potion_swap", "reward_button": 1, "discard_slot": 1}),
            json!({"action": "potion_swap", "reward_button": 1, "discard_slot": null}),
            json!({"action": "claim"}),
        ];
        let g = potion_swap_group(&opts);
        assert_eq!(g.len(), 2);
        assert!(g[0]["discard_slot"].is_null());
    }

    #[test]
    fn an_enchant_screen_without_its_enchantment_still_declines() {
        assert_eq!(
            screen_enchantment(&json!({"source_kind": "enchantment"})),
            None
        );
        assert_eq!(
            screen_enchantment(&json!({"source_model": "NIMBLE"})).as_deref(),
            Some("NIMBLE")
        );
        assert_eq!(
            screen_enchantment(&json!({"source_model": "X", "source_kind": "relic"})),
            None
        );
        // GH #574: the mod's field wins, and a present null is final.
        assert_eq!(
            screen_enchantment(&json!({"pending_enchantment": "GOOPY", "source_model": "X"}))
                .as_deref(),
            Some("GOOPY")
        );
        assert_eq!(
            screen_enchantment(&json!({"pending_enchantment": null, "source_model": "X"})),
            None
        );
        let (skill, attack) = (json!({"type": "Skill"}), json!({"type": "Attack"}));
        let power = json!({"type": "Power"});
        let some = |t: &str| Some(t.to_owned());
        assert_eq!(offered_card_types(&[&skill, &skill]), (some("skill"), None));
        assert_eq!(
            offered_card_types(&[&skill, &attack]),
            (some("skill"), some("attack"))
        );
        assert_eq!(offered_card_types(&[&skill, &attack, &power]), (None, None));
    }

    #[test]
    fn claws_takes_the_first_card_until_six_then_confirms() {
        let cards = [json!({"index": 3, "name": "Strike"})];
        let refs: Vec<&Value> = cards.iter().collect();
        let open = claws_rule(&json!({"max_select": 6, "selected_count": 2}), &refs);
        assert_eq!(
            open["best"],
            json!({"action": "choose", "index": 3, "label": "Strike"})
        );
        let full = claws_rule(&json!({"max_select": 6, "selected_count": 6}), &refs);
        assert_eq!(full["best"], json!({"action": "skip", "label": "confirm"}));
    }
}
