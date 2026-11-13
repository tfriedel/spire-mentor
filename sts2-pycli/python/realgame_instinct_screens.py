"""Instinct's out-of-combat screens: park the rebuilt run AT the screen, map back.

:mod:`realgame_instinct` rebuilds the live run from the mod's envelope
(:func:`realgame_instinct.graft_run`); this module puts that run on the screen
the player is looking at and turns the student's pick into the screen's own
option index.

**Inject, never re-roll.** The envelope's ``rng`` block carries every stream's
cursor but none of the history-dependent state (card rarity odds, unknown-room
odds, visited events, grab bags), so re-entering a room could roll a different
shop / event / reward than the screen shows. Each screen's options go into the
engine as shown, through the ``debug_*`` graft setters (``sts2_wire::graft``),
and the engine's offer is checked against the screen before the student's
answer is trusted.

**Map back by identity, not by index.** The engine's index spaces differ from
the screen's in ways that are easy to get subtly wrong (curse-first removal
order, upgradable-only upgrade lists, a shop's per-kind lists vs the screen's
flat one), so a deck pick is resolved by APPLYING it to a copy of the run and
seeing which card left the deck (removed, upgraded, transformed: the picked
card is the one missing afterwards), and a shop / rest / event / relic pick by
the item it names. A class that names nothing on the screen is dropped and the
crown passes to the student's next choice.

Every handler raises :class:`realgame_instinct.InstinctUnavailable` with a
player-facing reason when it cannot stand on the screen; the advisor then
answers with the search.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from display_names import card_name, display_name, potion_swap_label, shop_option_label
from realgame_instinct import (
    INSTINCT_GAME_VERSION,
    InstinctPick,
    InstinctUnavailable,
    graft_run,
    is_coop,
    note_for,
)

if TYPE_CHECKING:
    from realgame_instinct import InstinctAdvisor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def norm_id(raw: Any) -> str:
    """An id in any spelling (``STRIKE_IRONCLAD``, ``StrikeIronclad``,
    ``CARD.STRIKE_IRONCLAD``, ``RELIC.ANCHOR``) as one comparable key."""
    tail = str(raw or "").split(".")[-1]
    return re.sub(r"[^A-Z0-9]", "", tail.upper())


def card_spec(card: dict[str, Any]) -> dict[str, Any]:
    """A wire Card object as a graft card spec (``sts2_wire::graft``)."""
    level = card.get("upgrade_level")
    if level is None:
        level = 1 if card.get("upgraded") else 0
    spec: dict[str, Any] = {"id": card.get("id"), "upgraded": int(level or 0)}
    if card.get("enchantment"):
        spec["enchantment"] = card["enchantment"]
        spec["enchantment_amount"] = int(card.get("enchantment_amount") or 1)
    return spec


def _card_key(card_id: Any, upgraded: Any, enchantment: Any = None) -> tuple[str, bool, str]:
    return (norm_id(card_id), bool(upgraded), norm_id(enchantment) if enchantment else "")


def registry_id(raw: Any) -> str:
    """A wire card id in the engine registry's spelling (``SETUP_STRIKE`` ->
    ``SETUP_STRIKE_CARD``: a card whose name a power also has carries the
    suffix in the engine), so it compares with the engine's own names."""
    from sts2_hint.adapter import UnsupportedStateError
    from sts2_hint.adapter import registry_card_id as _registry

    try:
        return _registry(str(raw), "$.card")
    except (UnsupportedStateError, ValueError, KeyError):
        return str(raw or "")


def wire_card_key(card: dict[str, Any]) -> tuple[str, bool, str]:
    level = card.get("upgrade_level")
    upgraded = bool(level) if level is not None else bool(card.get("upgraded"))
    return _card_key(registry_id(card.get("id")), upgraded, card.get("enchantment"))


def deck_keys(rm: Any) -> list[tuple[str, bool, str]]:
    """The engine's master deck as identity keys, deck order."""
    detailed = list(rm.master_deck_detailed())
    try:
        enchants = list(rm.master_deck_enchantments())
    except Exception:
        enchants = [None] * len(detailed)
    out = []
    for i, (name, upgraded) in enumerate(detailed):
        ench = enchants[i] if i < len(enchants) else None
        if isinstance(ench, (list, tuple)):
            ench = ench[0] if ench else None
        elif isinstance(ench, dict):
            ench = ench.get("id") or ench.get("enchantment")
        out.append(_card_key(name, upgraded, ench))
    return out


def picked_deck_card(rm: Any, action: dict[str, Any]) -> tuple[str, bool, str] | None:
    """The deck card ``action`` takes away (removes / upgrades / transforms).

    Applied to a COPY of the run: the card in the deck before and not after
    is the pick, whatever index space the action uses.
    """
    before = Counter(deck_keys(rm))
    probe = rm.clone()
    try:
        probe.take_action(action)
    except Exception as exc:
        logger.debug("instinct: probing %s failed: %s", action, exc)
        return None
    gone = before - Counter(deck_keys(probe))
    if len(gone) != 1:
        return None
    return next(iter(gone))


def _find_card(
    cards: list[dict[str, Any]], key: tuple[str, bool, str], used: set[int]
) -> int | None:
    """The position of the first unused wire card with identity ``key``.

    Falls back to ignoring the enchantment when nothing matches exactly (the
    engine may not carry an enchantment the wire names, or vice versa).
    """
    for loose in (False, True):
        for i, c in enumerate(cards):
            if i in used:
                continue
            k = wire_card_key(c)
            if k == key or (loose and k[:2] == key[:2]):
                return i
    return None


def _label(c: dict[str, Any]) -> str:
    return str(c.get("name") or c.get("label") or c.get("id") or "?")


def screen_advice(
    pick: InstinctPick,
    rows: dict[int, tuple[dict[str, Any], str] | None],
    *,
    note_extra: str = "",
) -> dict[str, Any]:
    """The wire ``advice`` from per-class screen responses.

    ``rows[class] = (response, label)`` where ``response`` is the screen's
    ``{"action": "choose", "index": i}`` or ``{"action": "skip"}``; ``None``
    when the class names nothing on this screen (dropped). Classes that share
    a response (several removal targets answered by one "remove" button) are
    shown once, with their probabilities summed and the student's best
    member's label. The best is the student's greedy class, or, when that
    names nothing here, its next choice.
    """
    order = sorted(range(len(pick.probs)), key=lambda c: -pick.probs[c])
    merged: dict[tuple[Any, Any], dict[str, Any]] = {}
    best_key: tuple[Any, Any] | None = None
    for ci in order:
        row = rows.get(ci)
        if row is None:
            continue
        response, label = row
        key = (response.get("action"), response.get("index"))
        p = float(pick.probs[ci])
        if key not in merged:
            merged[key] = {"response": response, "label": label, "p": 0.0}
        merged[key]["p"] += p
        if best_key is None:
            best_key = key
    if best_key is None:
        raise InstinctUnavailable("none of Instinct's choices are on this screen")
    options = []
    for key, m in sorted(merged.items(), key=lambda kv: -kv[1]["p"]):
        resp = m["response"]
        options.append(
            {
                "label": m["label"],
                "visit_share": round(m["p"], 4),
                "prob": round(m["p"], 4),
                "is_best": key == best_key,
                "index": resp.get("index", -1),
                "action": resp.get("action", "choose"),
            }
        )
    best_row = merged[best_key]
    best = dict(best_row["response"])
    best["label"] = best_row["label"]
    note = note_for(pick, best_row["p"])
    if rows.get(pick.best) is None:
        note += " · its first choice is not on this screen"
    if note_extra:
        note += f" · {note_extra}"
    return {"note": note, "options": options, "best": best}


def lone_option(o: dict[str, Any]) -> dict[str, Any]:
    """A screen with one button: no decision, the button at probability 1."""
    label = str(o.get("label") or o.get("id") or "continue")
    index = int(o.get("index", 0))
    return {
        "note": "Instinct (e2e net, no search) · one option",
        "options": [
            {
                "label": label,
                "visit_share": 1.0,
                "prob": 1.0,
                "is_best": True,
                "index": index,
                "action": "choose",
            }
        ],
        "best": {"action": "choose", "index": index, "label": label},
    }


def _first_action(pick: InstinctPick, ci: int) -> dict[str, Any]:
    return (pick.decision["classes"][ci].get("actions") or [{}])[0] or {}


# ---------------------------------------------------------------------------
# Card offers
# ---------------------------------------------------------------------------


def card_reward(adv: InstinctAdvisor, state: dict[str, Any], session: Any) -> dict[str, Any]:
    if state.get("surface") == "choose_a_card" and state.get("combat") is not None:
        raise InstinctUnavailable("a card offered into the fight needs the fight mirror")
    cards = [c for c in (state.get("cards") or []) if isinstance(c, dict)]
    if not cards:
        raise InstinctUnavailable("no cards on this reward")
    rm, _notes = graft_run(state)
    rm.debug_set_room_type(adv.last_room or "monster")
    # Out of a fight, a choose-a-card screen adds to the deck (Lead
    # Paperweight, Hefty Tablet): the engine's run-effect choose screen
    # (`card_offer_choose_screen`), offered as the CARD_REWARD the student
    # saw in training. It declined these as "offered into the fight" before.
    rm.debug_open_card_offer(
        [card_spec(c) for c in cards],
        can_skip=bool(state.get("can_skip", True)),
        choose_screen=state.get("surface") == "choose_a_card",
    )
    offered = [norm_id(x) for x in rm.card_reward_option_ids()]
    if offered != [norm_id(registry_id(c.get("id"))) for c in cards]:
        raise InstinctUnavailable("the rebuilt reward does not match the screen")
    pick = adv.evaluate(state, rm, ("CARD_REWARD",))
    rows: dict[int, Any] = {}
    for ci in range(len(pick.probs)):
        a = _first_action(pick, ci)
        if a.get("action") != "pick_card_reward":
            rows[ci] = None  # Pael's Wing's sacrifice: not a button on this screen
            continue
        idx = a.get("idx")
        if idx is None:
            rows[ci] = ({"action": "skip"}, "skip")
        elif 0 <= int(idx) < len(cards):
            c = cards[int(idx)]
            rows[ci] = ({"action": "choose", "index": int(c.get("index", idx))}, _label(c))
        else:
            rows[ci] = None
    return screen_advice(pick, rows)


def card_bundle(adv: InstinctAdvisor, state: dict[str, Any], session: Any) -> dict[str, Any]:
    bundles = [b for b in (state.get("bundles") or []) if isinstance(b, dict)]
    if not bundles:
        raise InstinctUnavailable("no packs on this screen")
    rm, _notes = graft_run(state)
    rm.debug_open_card_offer(
        [], can_skip=False, bundles=[[card_spec(c) for c in b.get("cards") or []] for b in bundles]
    )
    pick = adv.evaluate(state, rm, ("CARD_BUNDLE",))
    rows: dict[int, Any] = {}
    for ci in range(len(pick.probs)):
        idx = _first_action(pick, ci).get("idx")
        if isinstance(idx, int) and 0 <= idx < len(bundles):
            b = bundles[idx]
            label = " + ".join(_label(c) for c in b.get("cards") or []) or f"pack {idx}"
            rows[ci] = ({"action": "choose", "index": int(b.get("index", idx))}, label)
        else:
            rows[ci] = None
    return screen_advice(pick, rows)


# ---------------------------------------------------------------------------
# Rest site + smith
# ---------------------------------------------------------------------------


def rest_site(adv: InstinctAdvisor, state: dict[str, Any], session: Any) -> dict[str, Any]:
    options = [o for o in (state.get("options") or []) if isinstance(o, dict)]
    if not options:
        raise InstinctUnavailable("no rest-site options")
    rm, _notes = graft_run(state)
    ids = [norm_id(x) for x in rm.debug_enter_rest_site()]
    by_id = {norm_id(o.get("id")): o for o in options if o.get("enabled", True)}
    pick = adv.evaluate(state, rm, ("REST",))
    rows: dict[int, Any] = {}
    for ci in range(len(pick.probs)):
        a = _first_action(pick, ci)
        idx = a.get("idx")
        o = by_id.get(ids[idx]) if isinstance(idx, int) and 0 <= idx < len(ids) else None
        rows[ci] = (
            ({"action": "choose", "index": int(o["index"])}, str(o.get("label") or o.get("id")))
            if o is not None
            else None
        )
    return screen_advice(pick, rows)


def _smith_rm(state: dict[str, Any]) -> Any:
    """The run standing on the smith's pick: rest site, Smith taken."""
    rm, _notes = graft_run(state)
    ids = [norm_id(x) for x in rm.debug_enter_rest_site()]
    if "SMITH" not in ids:
        raise InstinctUnavailable("the rebuilt rest site has no Smith")
    rm.take_action({"action": "rest_option", "idx": ids.index("SMITH")})
    return rm


# ---------------------------------------------------------------------------
# Deck screens (card_select out of combat)
# ---------------------------------------------------------------------------

#: prompt_key -> what the screen does (``realgame_ppo._PROMPT_SURFACES``' twin).
_PROMPTS = {
    "TO_REMOVE": "remove",
    "TO_TRANSFORM": "transform",
    "TO_ENCHANT": "enchant",
    "TO_UPGRADE": "upgrade",
    "TO_OBTAIN": "grant",
    "BRAIN_LEECH.PAGES.SHARE_KNOWLEDGE.SELECTIONSCREENPROMPT": "grant",
    "ROOM_FULL_OF_CHEESE.PAGES.GORGE.SELECTIONSCREENPROMPT": "grant",
    "SEA_GLASS.SELECTIONSCREENPROMPT": "grant",
    "CLAWS.SELECTIONSCREENPROMPT": "transform",
}


def _screen_enchantment(state: dict[str, Any]) -> str | None:
    """The enchantment an enchant screen applies (``source_model``), or ``None``.

    The mod names the first model argument of ``FromDeckForEnchantment``
    (``source_kind: "enchantment"``), e.g. ``NIMBLE`` or ``ADROIT``.
    """
    model = state.get("source_model")
    if not model or str(state.get("source_kind") or "enchantment").lower() != "enchantment":
        return None
    return str(model)


def _shared_card_type(cards: list[dict[str, Any]]) -> str | None:
    """The one card type every offered card has, when it restricts a pick.

    Self-Help Book enchants only one type (``c.Type == typeRestriction``).
    Restricting the engine's pool to the screen's shared type never offers a
    card the screen does not show.
    """
    types = {str(c.get("type") or "").lower() for c in cards}
    if len(types) == 1 and (t := types.pop()) in ("attack", "skill", "power"):
        return t
    return None


def deck_select(adv: InstinctAdvisor, state: dict[str, Any], session: Any) -> dict[str, Any]:
    cards = [c for c in (state.get("cards") or []) if isinstance(c, dict)]
    if not cards:
        raise InstinctUnavailable("no cards on this screen")
    kind = _PROMPTS.get(str(state.get("prompt_key") or "").upper())
    if kind is None:
        raise InstinctUnavailable("Instinct does not know what this card screen does")
    n = max(1, int(state.get("max_select") or state.get("max_count") or 1))
    lo = int(state.get("min_select") or state.get("min_count") or 0)

    if kind == "remove" and adv.last_screen == "shop" and adv.shop_removal is not None:
        # The shop already chose WHICH card (the engine's removal lives inside
        # the shop decision); this screen only executes it.
        i = _find_card(cards, adv.shop_removal, set())
        if i is not None:
            c = cards[i]
            return {
                "note": "Instinct (e2e net, no search) · the card it picked in the shop",
                "options": [
                    {
                        "label": _label(x),
                        "is_best": j == i,
                        "index": int(x.get("index", j)),
                        "action": "choose",
                    }
                    for j, x in enumerate(cards)
                ],
                "best": {"action": "choose", "index": int(c.get("index", i)), "label": _label(c)},
            }

    if kind == "grant":
        rm, _notes = graft_run(state)
        rm.debug_queue_card_grant([card_spec(c) for c in cards], pick=n, min_pick=lo)
        pick = adv.evaluate(state, rm, ("CARD_GRANT",))
        rows: dict[int, Any] = {}
        for ci in range(len(pick.probs)):
            a = _first_action(pick, ci)
            idx = a.get("idx")
            if a.get("action") == "grant_deck_card" and isinstance(idx, int) and idx < len(cards):
                c = cards[idx]
                rows[ci] = ({"action": "choose", "index": int(c.get("index", idx))}, _label(c))
            elif "confirm" in str(a.get("action")) or "skip" in str(a.get("action")):
                rows[ci] = ({"action": "skip"}, "take none")
            else:
                rows[ci] = None
        return screen_advice(pick, rows)

    if kind == "enchant":
        enchantment = _screen_enchantment(state)
        if enchantment is None:
            raise InstinctUnavailable("the screen does not say which enchantment it applies")
    if kind == "upgrade" and adv.last_screen == "rest_site":
        rm = _smith_rm(state)
        expect: tuple[str, ...] = ("SMITH",)
    else:
        rm, _notes = graft_run(state)
        if kind == "remove":
            rm.debug_queue_deck_removals(n)
            expect = ("CARD_REMOVE",)
        elif kind == "upgrade":
            rm.debug_queue_deck_upgrades(n)
            expect = ("CARD_UPGRADE",)
        elif kind == "enchant":
            try:
                # The net reads only the enchantment (``tokens.rs`` OFFER), so
                # the amount the screen does not carry changes nothing here.
                rm.debug_queue_deck_enchant(
                    enchantment,
                    optional=lo == 0,
                    count=n,
                    card_type=_shared_card_type(cards),
                )
            except ValueError as exc:
                raise InstinctUnavailable(f"the engine has no enchantment {enchantment}") from exc
            expect = ("CARD_ENCHANT",)
        else:
            rm.debug_queue_deck_transforms(n)
            expect = ("CARD_TRANSFORM",)
    pick = adv.evaluate(state, rm, expect)
    rows = {}
    for ci in range(len(pick.probs)):
        a = _first_action(pick, ci)
        key = picked_deck_card(rm, a)
        i = _find_card(cards, key, set()) if key is not None else None
        if i is None:
            if "skip" in str(a.get("action")) or "decline" in str(a.get("action")):
                rows[ci] = ({"action": "skip"}, "none")
            else:
                rows[ci] = None
            continue
        c = cards[i]
        rows[ci] = ({"action": "choose", "index": int(c.get("index", i))}, _label(c))
    return screen_advice(pick, rows)


# ---------------------------------------------------------------------------
# Shop
# ---------------------------------------------------------------------------


def shop(adv: InstinctAdvisor, state: dict[str, Any], session: Any) -> dict[str, Any]:
    options = [o for o in (state.get("options") or []) if isinstance(o, dict)]
    if not options:
        raise InstinctUnavailable("no shop options")
    usable = [o for o in options if o.get("enabled", True)]
    by_kind: dict[str, list[dict[str, Any]]] = {"buy_card": [], "buy_relic": [], "buy_potion": []}
    for o in usable:
        if o.get("action") in by_kind:
            by_kind[o["action"]].append(o)
    remove_opt = next((o for o in usable if o.get("action") == "remove_card"), None)
    leave_opt = next((o for o in options if o.get("action") == "leave_shop"), None)
    removal_cost = state.get("purge_cost")
    if removal_cost is None and remove_opt is not None:
        removal_cost = remove_opt.get("price", 0)
    inventory = {
        "cards": [
            {
                "id": (o.get("item") or {}).get("id"),
                "upgraded": bool((o.get("item") or {}).get("upgrade_level")),
                "price": int(o.get("price") or 0),
                "on_sale": False,
                "is_colorless": False,
                "available": True,
            }
            for o in by_kind["buy_card"]
        ],
        "relics": [
            {
                "id": (o.get("item") or {}).get("id"),
                "price": int(o.get("price") or 0),
                "available": True,
            }
            for o in by_kind["buy_relic"]
        ],
        "potions": [
            {
                "id": (o.get("item") or {}).get("id"),
                "price": int(o.get("price") or 0),
                "available": True,
            }
            for o in by_kind["buy_potion"]
        ],
        "removal_cost": int(removal_cost or 0),
        "removal_used": remove_opt is None,
    }
    rm, _notes = graft_run(state)
    try:
        unmatched = rm.debug_open_shop(inventory)
    except ValueError as exc:
        raise InstinctUnavailable(f"the engine cannot stock this shop ({exc})") from exc
    if unmatched:
        logger.debug("instinct: shop prices kept as shown: %s", unmatched)
    pick = adv.evaluate(state, rm, ("SHOP",))
    deck = deck_keys(rm)
    kinds = {
        "shop_buy_card": "buy_card",
        "shop_buy_relic": "buy_relic",
        "shop_buy_potion": "buy_potion",
    }
    rows: dict[int, Any] = {}
    removal_of: dict[int, tuple[str, bool, str]] = {}
    for ci in range(len(pick.probs)):
        a = _first_action(pick, ci)
        name = a.get("action")
        if name in kinds:
            lst = by_kind[kinds[name]]
            idx = a.get("idx")
            o = lst[idx] if isinstance(idx, int) and 0 <= idx < len(lst) else None
            rows[ci] = (
                ({"action": "choose", "index": int(o["index"])}, shop_option_label(o))
                if o is not None
                else None
            )
        elif name == "shop_remove_card" and remove_opt is not None:
            d = a.get("deck_idx")
            if isinstance(d, int) and 0 <= d < len(deck):
                removal_of[ci] = deck[d]
                rows[ci] = (
                    {"action": "choose", "index": int(remove_opt["index"])},
                    f"Remove {_deck_card_name(state, deck[d])}",
                )
            else:
                rows[ci] = None
        elif name == "leave_shop" and leave_opt is not None:
            rows[ci] = ({"action": "choose", "index": int(leave_opt["index"])}, "Leave shop")
        else:
            rows[ci] = None
    advice = screen_advice(pick, rows)
    # Carry the student's removal target to the follow-up card screen.
    order = sorted(range(len(pick.probs)), key=lambda c: -pick.probs[c])
    first_on_screen = next((c for c in order if rows.get(c) is not None), None)
    adv.shop_removal = removal_of.get(first_on_screen) if first_on_screen is not None else None
    return advice


def _deck_card_name(state: dict[str, Any], key: tuple[str, bool, str]) -> str:
    """An engine deck card (identity ``key``) by the name the run's wire
    deck gives it, else its id's display name."""
    deck = [
        c for c in (state.get("run_state") or {}).get("master_deck") or [] if isinstance(c, dict)
    ]
    i = _find_card(deck, key, set())
    if i is not None:
        return card_name(deck[i])
    return display_name(key[0]) + ("+" if key[1] else "")


# ---------------------------------------------------------------------------
# Events (Neow included) and relic choices
# ---------------------------------------------------------------------------

#: Event pages are reached by replaying earlier picks from page 0; bound the
#: search (the deepest live event is a few pages).
_EVENT_REPLAY_DEPTH = 3


def _tail(text_key: Any) -> str:
    return norm_id(str(text_key or "").split(".")[-1])


def _engine_event_tails(rm: Any) -> list[str]:
    infos = rm.event_option_infos() or []
    # The loc key is the game's own option name (the screen's `text_key`); the
    # engine's `option_id` is its own shorthand (`nab_map` for NAB_THE_MAP).
    return [_tail(i.get("loc_key")) or norm_id(i.get("option_id")) for i in infos]


def _is_subsequence(want: list[str], have: list[str]) -> bool:
    it = iter(have)
    return all(any(w == h for h in it) for w in want)


def _find_event_page(rm: Any, want: list[str]) -> Any:
    """``rm`` (page 0) or a replayed copy showing the screen's options.

    The screen lists only the buttons the game unlocked (PROTOCOL.md
    ``event``), the engine every option of the page, a locked one included
    (Self-Help Book's third read): the screen's option names must appear among
    the page's, in order.
    """
    frontier = [rm]
    for _depth in range(_EVENT_REPLAY_DEPTH + 1):
        nxt = []
        for cand in frontier:
            if want and _is_subsequence(want, _engine_event_tails(cand)):
                return cand
            for i in range(len(cand.event_option_infos() or [])):
                probe = cand.clone()
                try:
                    probe.take_action({"action": "event_choice", "idx": i})
                except Exception:
                    continue
                if probe.phase() == "event":
                    nxt.append(probe)
        frontier = nxt
    return None


def event(adv: InstinctAdvisor, state: dict[str, Any], session: Any) -> dict[str, Any]:
    import sts2_rs

    options = [o for o in (state.get("options") or []) if isinstance(o, dict)]
    if not options:
        raise InstinctUnavailable("no event options")
    enabled = [o for o in options if o.get("enabled", True)]
    if len(enabled) == 1:
        return lone_option(enabled[0])
    event_id = str(options[0].get("event_id") or "")
    want = [_tail(o.get("text_key")) for o in options]
    if norm_id(event_id) == "NEOW":
        snap_rs = state.get("run_state") or {}
        rm = sts2_rs.RunManager(
            game_version=INSTINCT_GAME_VERSION,
            string_seed=str(state.get("seed")),
            character="Ironclad",
            ascension_level=int(snap_rs.get("ascension") or 0),
            neow_start=True,
            mode="agent",
            interactive_potion_swap=True,
            act1="roll",
        )
        if is_coop(state):
            rm.debug_use_coop_maps()
        expect: tuple[str, ...] = ("NEOW",)
    else:
        if all(o.get("relic_id") for o in options if o.get("enabled", True)):
            return _ancient_relic_choice(adv, state, options)
        # A later page is replayed from the event's FIRST page: the envelope
        # of a later one already carries the earlier picks' HP / gold / cards,
        # and replaying them on top of it would count them twice.
        track = _event_track(adv, state, event_id, [_tail(o.get("text_key")) for o in options])
        rm, _notes = graft_run(track["origin"])
        try:
            rm.debug_enter_event(event_id)
        except ValueError as exc:
            raise InstinctUnavailable(f"the engine has no event {event_id}") from exc
        expect = ("EVENT",)
    if expect == ("NEOW",):
        # Neow's engine options are slots (`positive_1`, `cursed`), its offer
        # the relic behind each (`neow_offers()`); the screen names the relic.
        by_relic = {norm_id(o.get("relic_id")): o for o in options if o.get("relic_id")}
        offers = [norm_id(r) for r in rm.neow_offers()]
        slot_option = [by_relic.get(r) for r in offers]
        relic_ids = [str(o.get("relic_id") or "") for o in options]
        if (
            sum(o is not None for o in slot_option) != len(by_relic)
            and len(options) == 3
            and all(relic_ids)
        ):
            # The engine rolls single player's Neow. Co-op seeds each
            # player's with their slot and keeps Massive Scroll in the pool,
            # so the roll differs there: stand on the screen's own three
            # offers (the game lists the two positive ones, then the curse).
            try:
                rm.debug_set_neow_offers(*relic_ids)
            except ValueError as exc:
                raise InstinctUnavailable(
                    f"the engine has no Neow offer like this ({exc})"
                ) from exc
            offers = [norm_id(r) for r in rm.neow_offers()]
            slot_option = [by_relic.get(r) for r in offers]
        if sum(o is not None for o in slot_option) != len(by_relic) or not by_relic:
            raise InstinctUnavailable("the rebuilt Neow offers other relics than the screen")
        page = rm
    else:
        page = _replay_trail(rm, track["trail"], want)
        if page is None:
            page = _find_event_page(rm, want)
        if page is None:
            raise InstinctUnavailable("the rebuilt event does not show this page")
        slot_option = _match_options(_engine_event_tails(page), options)
    pick = adv.evaluate(state, page, expect)
    rows: dict[int, Any] = {}
    for ci in range(len(pick.probs)):
        idx = _first_action(pick, ci).get("idx")
        o = slot_option[idx] if isinstance(idx, int) and 0 <= idx < len(slot_option) else None
        if o is None or not o.get("enabled", True):
            rows[ci] = None
            continue
        rows[ci] = (
            {"action": "choose", "index": int(o.get("index", idx))},
            str(o.get("label") or "?"),
        )
    advice = screen_advice(pick, rows)
    if expect == ("EVENT",):
        # What this page's buttons are in the engine, and Instinct's own pick:
        # the next page of this event replays whichever was taken.
        wire_to_engine = {
            int(o.get("index", i)): i for i, o in enumerate(slot_option) if o is not None
        }
        best_index = (advice.get("best") or {}).get("index")
        adv.event_track["pending"] = {
            "sig": tuple(want),
            "wire_to_engine": wire_to_engine,
            "best": wire_to_engine.get(best_index) if isinstance(best_index, int) else None,
            "human": None,
        }
    return advice


def _loose(tail: str) -> str:
    """A tail without its page number (`EXTRACT_CURRENT_PRIZE_2` -> `..._PRIZE`):
    some engine events keep page 1's loc keys on every page."""
    return re.sub(r"\d+$", "", tail)


def _event_track(
    adv: InstinctAdvisor, state: dict[str, Any], event_id: str, want: list[str]
) -> dict[str, Any]:
    """This event's trail: the engine option taken on each earlier page.

    One per event visit, keyed by seed, floor and event. When a request shows a
    different page than the one last answered, the option taken there is the
    player's (``human_action``, :func:`note_event_choice`) or, when the AI
    played it, Instinct's own pick.
    """
    key = (state.get("seed"), state.get("floor"), norm_id(event_id))
    track = adv.event_track
    if track is None or track.get("key") != key:
        track = {"key": key, "trail": [], "pending": None, "origin": state}
        adv.event_track = track
    pending = track.get("pending")
    if pending is not None and pending["sig"] != tuple(want):
        chosen = pending["human"] if pending["human"] is not None else pending["best"]
        if chosen is not None:
            track["trail"].append(chosen)
        track["pending"] = None
    return track


def note_event_choice(adv: InstinctAdvisor, msg: dict[str, Any]) -> None:
    """A ``human_action`` on an event screen: the page's taken option."""
    track = adv.event_track
    choice = msg.get("choice") or {}
    if track is None or track.get("pending") is None or not isinstance(choice.get("index"), int):
        return
    track["pending"]["human"] = track["pending"]["wire_to_engine"].get(choice["index"])


def _replay_trail(rm: Any, trail: list[int], want: list[str]) -> Any:
    """``rm`` replayed through ``trail`` from page 0, if it shows the screen."""
    if not trail:
        return None
    probe = rm.clone()
    for idx in trail:
        try:
            probe.take_action({"action": "event_choice", "idx": int(idx)})
        except Exception:
            return None
        if probe.phase() != "event":
            return None
    have = [_loose(t) for t in _engine_event_tails(probe)]
    if _is_subsequence([_loose(t) for t in want], have):
        return probe
    # The trail is the stronger evidence: some engine events keep page 1's
    # option names on later pages (Colossal Flower), so a page the trail
    # reaches with as many options as the screen is taken as the screen.
    return probe if len(have) == len(want) else None


def _match_options(tails: list[str], options: list[dict[str, Any]]) -> list[Any]:
    """Engine option i -> the screen's button for it (``None`` when the screen
    leaves it out, locked): by exact name, else by name without its page
    number, else by position among the unmatched."""
    exact = {_tail(o.get("text_key")): o for o in options}
    loose = {_loose(_tail(o.get("text_key"))): o for o in options}
    out: list[Any] = [exact.get(t) or loose.get(_loose(t)) for t in tails]
    if not any(o is not None for o in out) and len(tails) == len(options):
        return list(options)  # no name in common: the same buttons, in order
    return out


def _ancient_relic_choice(
    adv: InstinctAdvisor, state: dict[str, Any], options: list[dict[str, Any]]
) -> dict[str, Any]:
    """An Act 2 / 3 Ancient's relic offer: the engine's boss-relic pick."""
    offered = [o for o in options if o.get("enabled", True) and o.get("relic_id")]
    rm, _notes = graft_run(state)
    try:
        rm.debug_open_boss_relic([str(o["relic_id"]) for o in offered])
    except ValueError as exc:
        raise InstinctUnavailable(f"the engine cannot offer these relics ({exc})") from exc
    return _relic_rows(adv, state, rm, offered, lambda o: str(o.get("label") or o.get("relic_id")))


def boss_relic(adv: InstinctAdvisor, state: dict[str, Any], session: Any) -> dict[str, Any]:
    offered = [
        o for o in (state.get("options") or []) if isinstance(o, dict) and o.get("enabled", True)
    ]
    ids = [str((o.get("relic") or {}).get("id") or "") for o in offered]
    if not offered or not all(ids):
        raise InstinctUnavailable("no relics on this screen")
    rm, _notes = graft_run(state)
    try:
        rm.debug_open_boss_relic(ids)
    except ValueError as exc:
        raise InstinctUnavailable(f"the engine cannot offer these relics ({exc})") from exc
    return _relic_rows(
        adv, state, rm, offered, lambda o: str((o.get("relic") or {}).get("name") or o.get("label"))
    )


def _relic_rows(
    adv: InstinctAdvisor, state: dict[str, Any], rm: Any, offered: list[dict[str, Any]], label: Any
) -> dict[str, Any]:
    pick = adv.evaluate(state, rm, ("BOSS_RELIC",))
    rows: dict[int, Any] = {}
    for ci in range(len(pick.probs)):
        idx = _first_action(pick, ci).get("idx")
        if isinstance(idx, int) and 0 <= idx < len(offered):
            o = offered[idx]
            rows[ci] = ({"action": "choose", "index": int(o.get("index", idx))}, label(o))
        else:
            rows[ci] = None
    return screen_advice(pick, rows)


# ---------------------------------------------------------------------------
# Full-belt potion swap
# ---------------------------------------------------------------------------


def potion_swap(adv: InstinctAdvisor, state: dict[str, Any], session: Any) -> dict[str, Any]:
    import sts2_realgame

    group = sts2_realgame.potion_swap_group(state.get("options") or [])
    if not group:
        raise InstinctUnavailable("no potion swap on this screen")
    potion = str((group[0].get("reward") or {}).get("id") or "")
    if not potion:
        raise InstinctUnavailable("the screen does not name the potion")
    rm, _notes = graft_run(state)
    rm.debug_queue_potion_swap(potion)
    pick = adv.evaluate(state, rm, ("POTION_SWAP",))
    by_slot = {o.get("discard_slot"): o for o in group}
    belt = (state.get("run_state") or {}).get("potions") or state.get("potions") or []
    rows: dict[int, Any] = {}
    for ci in range(len(pick.probs)):
        a = _first_action(pick, ci)
        o = by_slot.get(a.get("discard_slot")) if "discard_slot" in a else None
        rows[ci] = (
            ({"action": "choose", "index": int(o["index"])}, potion_swap_label(o, belt))
            if o is not None
            else None
        )
    advice = screen_advice(pick, rows)
    # `best.label` stays the mod's own spelling (`discard <POTION_ID>`): the
    # takeover cross-checks it (SimAdvisor.ReconcileIndex).
    chosen = next((o for o in group if o.get("index") == advice["best"].get("index")), None)
    if chosen is not None and chosen.get("label"):
        advice["best"]["label"] = str(chosen["label"])
    return advice


#: ``advice_for`` -> handler.
HANDLERS = {
    "card_reward": card_reward,
    "card_bundle": card_bundle,
    "rest_site": rest_site,
    "card_select": deck_select,
    "shop": shop,
    "event": event,
    "boss_relic": boss_relic,
    "potion_swap": potion_swap,
}
