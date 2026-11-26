"""Instinct (the e2e student, no search) on referee observations.

Instinct reads an engine run standing at the decision. The live advisor
rebuilds it from the mod's message with the run's seed; a referee observation
has none, so every request here is ``seedless``: the run is rebuilt from the
act map and the bosses the screen shows (``sts2_advisor::graft``), and the
student reads nothing a player could not see.

Two things the student was trained on are not on the screen and are filled in
here from what is:

* the fight's encounter id: the encounter whose roster, act and room tier
  match the fight's opening enemies (unique in the engine's tables);
* the act-1 variant (Overgrowth or Underdocks): the context's ``act_name``.

Event outcome previews are resolved on the rebuilt run's dice, which are not
the referee's: what the student sees there is a plausible preview, not the
real one.
"""

from __future__ import annotations

import functools
import logging
from collections import Counter
from typing import Any

import sts2_rs
from ppo import vocab
from realgame_instinct import InstinctAdvisor, InstinctUnavailable

logger = logging.getLogger(__name__)

_TIERS = {"monster": ("Weak", "Normal"), "elite": ("Elite",), "boss": ("Boss",)}


def _norm_act(name: str) -> str:
    return name.strip().lower()


def _event_option(option: dict[str, Any]) -> dict[str, Any]:
    """A referee event button in the mod's terms.

    The text key (``NEOW.pages.INITIAL.options.LARGE_CAPSULE``) names the
    event and the option. The event id comes from it, not from the event's
    display name (Reflections shows its own mirrored). A relic-granting
    button names its relic in ``relic_id``, as the mod does: Neow's and the
    Ancients' offers are rolled, so the rebuild stands on the screen's. The
    Doll Room's buttons are the relics themselves, keyed by their names.
    """
    key = str(option.get("text_key") or "")
    if ".pages." in key:
        tail = key.rsplit(".", 1)[-1]
        relic = tail if _is_relic(tail) else ""
        out = {**option, "relic_id": relic, "event_id": key.split(".", 1)[0]}
    else:
        out = {**option, "relic_id": _relic_named(key)}
    return out


def _is_relic(name: str) -> bool:
    reg = vocab.relics()
    return bool(name) and reg.index(name) != reg.oov


@functools.cache
def _relics_by_name() -> dict[str, str]:
    import display_names

    return {display_names.display_name(r).lower(): r for r in vocab.relics().names}


def _relic_named(name: str) -> str:
    """The relic id a display name stands for, or ``""``."""
    return _relics_by_name().get(name.strip().lower(), "")


def infer_encounter(act_name: str, room_type: str, kinds: list[str]) -> str | None:
    """The engine encounter a fight is, from what the screen shows.

    Args:
        act_name: The context's ``act_name`` (``"Underdocks"``, ``"Hive"``).
        room_type: ``monster`` / ``elite`` / ``boss``.
        kinds: The fight's opening enemies, as engine kinds.

    Returns:
        The id when exactly one encounter of that act and tier has that roster,
        else None (an event fight, or a roster the tables do not hold).
    """
    tiers = _TIERS.get(room_type.lower())
    if tiers is None:
        return None
    # Match every roster the encounter can roll, not its canonical one: the
    # slime fights pick their members at spawn, so the screen's roster is
    # usually not the listed one (GH #1659).
    want = sorted(kinds)
    hits = [
        e["id"]
        for e in sts2_rs.list_encounters()
        if _norm_act(e["act_name"]) == _norm_act(act_name)
        and e["tier"] in tiers
        and want in e["rosters"]
    ]
    return hits[0] if len(hits) == 1 else None


#: The act-start Ancients, whose relic offer is the act's boss relic.
_ANCIENTS = frozenset({"OROBAS", "PAEL", "TEZCATARA", "NONUPEIPE", "TANX", "VAKUU", "DARV"})


def _is_boss_relic_offer(bridge_state: dict[str, Any]) -> bool:
    """An act's boss-relic offer: the host's rerouted ``boss_relic`` screen,
    or an act-start Ancient's event page (GH #1678)."""
    if bridge_state.get("type") == "boss_relic":
        return True
    if bridge_state.get("type") != "event":
        return False
    keys = [str(o.get("text_key") or "") for o in bridge_state.get("options") or []]
    return bool(keys) and all(k.split(".", 1)[0] in _ANCIENTS for k in keys)


# The events whose button starts a fight, and the encounter the engine names
# it: after the event, except The Lantern Key's Mysterious Knight
# (`TheLanternKey.cs`: `EnterCombatWithoutExitingEvent<MysteriousKnightEventEncounter>`).
_EVENT_FIGHTS = {
    "BATTLEWORN_DUMMY": "BATTLEWORN_DUMMY",
    "DENSE_VEGETATION": "DENSE_VEGETATION",
    "FAKE_MERCHANT": "FAKE_MERCHANT",
    "THE_LANTERN_KEY": "MYSTERIOUS_KNIGHT",
    "PUNCH_OFF": "PUNCH_OFF",
    "THE_ARCHITECT": "THE_ARCHITECT",
}


def event_encounter(text_key: str, game_version: str) -> str | None:
    """The encounter an event button starts, as the engine names it, or None.

    ``text_key`` is the pressed button's (``PUNCH_OFF.pages.INITIAL.options.FIGHT``).
    The engine names an event's fight after the event; from v0.111.0 the
    Battleworn Dummy has one per setting (``event_encounter_id_for``).
    Whether the button starts a fight at all is the caller's to know: this
    is read only for a fight that follows it.
    """
    event = text_key.split(".", 1)[0]
    if event not in _EVENT_FIGHTS:
        return None
    setting = text_key.rsplit(".", 1)[-1]
    if (
        event == "BATTLEWORN_DUMMY"
        and game_version != "v0.107.1"
        and setting.startswith("SETTING_")
    ):
        return f"BATTLEWORN_DUMMY_EVENT_V{setting.removeprefix('SETTING_')}_ENCOUNTER"
    return f"{_EVENT_FIGHTS[event]}_EVENT_ENCOUNTER"


def _fight_tier(state: dict[str, Any], room_type: str) -> str:
    """The fight's tier: the game's room (``Monster``/``Elite``/``Boss``) first.

    The caller's ``room_type`` is the map node's, ``unknown`` for a ``?``
    that rolled a fight, which names no tier (GH #1659's encounter id then
    stayed 0)."""
    for r in ((state.get("context") or {}).get("room_type"), room_type):
        if isinstance(r, str) and r.lower() in _TIERS:
            return r.lower()
    return "monster"


def _choose_a_card(msg: dict[str, Any]) -> dict[str, Any]:
    """An out-of-fight choose-a-card screen (Lead Paperweight) as the mod sends it.

    The host reports it as a ``card_select``; the mod as a ``card_reward``
    with ``surface: choose_a_card``, the CARD_REWARD the student was trained
    on (GH #1665). ``cancelable`` is the game's ``canSkip``.
    """
    out = {k: v for k, v in msg.items() if k not in ("min_count", "max_count", "cancelable")}
    return {
        **out,
        "type": "card_reward",
        "surface": "choose_a_card",
        "can_skip": bool(msg.get("cancelable")),
    }


# Relics that heal after the player's turn starts (`AfterPlayerTurnStartLate`),
# so before a fight's first decision; the engine's run HP is from before.
_TURN_START_HEAL = {"BLOOD_VIAL": 2, "FAKE_BLOOD_VIAL": 1}
#: Royal Poison hits at the first turn's start (`AfterPlayerTurnStart`, turn 1),
#: before those heals, so before the first decision too (GH #1831).
_TURN_START_DAMAGE = {"ROYAL_POISON": 4}


#: Pantograph's heal on entering a boss fight (`Pantograph.BeforeCombatStart`).
_PANTOGRAPH_HEAL = 25
#: Planisphere heals entering a `?` room (`Planisphere.AfterRoomEntered`).
_PLANISPHERE_HEAL = 5


def _fight_start_hp(
    run_state: dict[str, Any],
    room_hp: int | None,
    *,
    healed: bool,
    boss: bool = False,
    unknown: bool = False,
    from_event: bool = False,
) -> dict[str, Any]:
    """The run's HP as the fight started, read at its first decision (GH #1705)."""
    hp, max_hp = run_state.get("hp"), run_state.get("max_hp")
    # A melted wax relic (Toy Box) does nothing any more: a melted Blood
    # Vial heals no one (Spire Bench seeds 180000158/258/739/1873).
    ids = [
        str((r or {}).get("id") or "").rsplit(".", 1)[-1]
        for r in run_state.get("relics") or []
        if not (r or {}).get("melted")
    ]
    # An Imbued card plays itself at the first turn's start
    # (`Imbued.AfterAutoPrePlayPhaseEntered`), before the first decision, and
    # what it plays is unknowable from here (an Imbued Havoc plays the draw
    # pile's top card): Offering's 6 HP (seeds 180000474, 180000979).
    imbued = any(
        (c or {}).get("enchantment") == "Imbued" for c in run_state.get("master_deck") or []
    )
    earring = healed and ("WHISPERING_EARRING" in ids or imbued)
    # What heals between the screen before the fight and its first turn: a
    # boss room's Pantograph, a `?` room's Planisphere. A fight an event
    # button started has the event's own page before it, which already shows
    # Planisphere's heal; and no such button moves HP itself (v0.111.0
    # `EnterCombatWithoutExitingEvent` callers: Battleworn Dummy, Dense
    # Vegetation's FIGHT after its REST page, Fake Merchant, The Lantern Key,
    # Punch Off, The Architect).
    entry = (
        0
        if from_event
        else (_PANTOGRAPH_HEAL if boss and "PANTOGRAPH" in ids else 0)
        + (_PLANISPHERE_HEAL if unknown and "PLANISPHERE" in ids else 0)
    )
    if earring and room_hp is not None and isinstance(max_hp, int):
        # Whispering Earring (or an Imbued card) played before this decision
        # (`AfterAutoPrePlayPhaseEnteredLate`), so the HP here is already the
        # fight's. The fight started on the HP of the screen before it, plus
        # a boss room's Pantograph (GH #1751); an event's fight on its page's
        # (Battleworn Dummy: seeds 180001585, 180001606).
        return {"hp": min(max_hp, room_hp + entry), "max_hp": max_hp}
    heal = sum(_TURN_START_HEAL.get(i, 0) for i in ids) if healed else 0
    loss = sum(_TURN_START_DAMAGE.get(i, 0) for i in ids) if healed else 0
    if loss and "TUNGSTEN_ROD" in ids:
        loss -= 1  # each HP loss is one less (`TungstenRod.ModifyHpLostAfterOsty`)
    if loss and isinstance(hp, int) and isinstance(max_hp, int):
        # The hit came first and made room, so no heal after it was capped.
        return {"hp": min(max_hp, hp + loss - heal), "max_hp": max_hp}
    if not heal or not isinstance(hp, int) or not isinstance(max_hp, int):
        return {"hp": hp, "max_hp": max_hp}
    if hp < max_hp:
        return {"hp": hp - heal, "max_hp": max_hp}
    # At full HP the heal may have been capped; the screen before the fight,
    # plus what healed on entering the room, bounds it.
    floor = max_hp - heal
    return {
        "hp": max(floor, min(room_hp + entry, max_hp)) if room_hp is not None else floor,
        "max_hp": max_hp,
    }


class InstinctPolicy:
    """Answers referee decisions with Instinct; None where it cannot."""

    def __init__(self, character: str, ascension: int, game_version: str) -> None:
        self.character = character
        self.ascension = ascension
        self.game_version = game_version
        self.adv = InstinctAdvisor()
        self.encounter: str | None = None
        self.fight_hp: dict[str, Any] | None = None
        # The player's HP on the last screen outside a fight (GH #1705).
        self.room_hp: int | None = None
        # The floor of the fight in progress, None outside one (GH #1752).
        self.fight_floor: int | None = None
        self.last_room_type: str | None = None
        # The floor of the last screen that named its room (GH #1704).
        self.last_room_floor: int | None = None
        # The event button last pressed, while no other screen followed it:
        # a fight that starts from it is that event's (GH #1676).
        self.event_button: str | None = None
        # The fight an event button started: the game names its room a
        # `Monster` room, the engine stands in the event's (GH #1688).
        self.event_fight = False
        # The last screen's view (its state and map), and the previous act's
        # last one: the act's end, once the next act's Ancient offers the boss
        # relic (GH #1678), after its dialogue pages.
        self.last_view: tuple[dict[str, Any], dict[str, Any] | None] | None = None
        self.act_end_view: tuple[dict[str, Any], dict[str, Any] | None] | None = None
        self.unavailable: Counter[str] = Counter()

    def new_run(self) -> None:
        """Forget the last run: its screens, its fight, its misses."""
        self.adv.reset("new run")
        self.encounter = None
        self.fight_hp = None
        self.room_hp = None
        self.fight_floor = None
        self.last_room_type = None
        self.last_room_floor = None
        self.event_button = None
        self.event_fight = False
        self.last_view = None
        self.act_end_view = None
        self.unavailable = Counter()

    # -- the rebuilt run ---------------------------------------------------

    def saw(self, state: dict[str, Any]) -> None:
        """Note the room a screen stands in, asked of us or not.

        The host answers some screens itself (a treasure chest's reward
        screen); the run still stands in that room at the next map, so the
        game wrappers show those screens here too (GH #1704).
        """
        player = state.get("player") or {}
        decision = state.get("decision")
        in_fight = decision == "combat_play" or (
            decision == "card_select" and state.get("combat") is not None
        )
        if not in_fight:
            self.fight_floor = None
            if isinstance(player.get("hp"), int):
                self.room_hp = player["hp"]
        ctx = state.get("context") or {}
        room = ctx.get("room_type")
        floor = ctx.get("total_floor")
        if room == "Map":
            self.event_fight = False
            # A map on a later floor than the last room's screen: the room
            # between showed none. Only an empty chest does that (Silver
            # Crucible's first, `SilverCrucible.ShouldGenerateTreasure`), so
            # the run stands in a treasure room, whatever the `?` read
            # (Spire Bench seed 180000315).
            if (
                isinstance(floor, int)
                and self.last_room_floor is not None
                and floor > self.last_room_floor
            ):
                self.last_room_type = "Treasure"
                self.last_room_floor = floor
        elif room:
            self.last_room_type = "Event" if self.event_fight and room == "Monster" else str(room)
            if isinstance(floor, int):
                self.last_room_floor = floor

    def envelope(self, state: dict[str, Any], bridge_map: dict[str, Any] | None) -> dict[str, Any]:
        """The seedless fields a request needs on top of the screen's envelope."""
        ctx = state.get("context") or {}
        act = int(ctx.get("act") or 1)
        out: dict[str, Any] = {
            "seedless": True,
            "game_version": self.game_version,
            "act": act,
            "floor": int(ctx.get("total_floor", ctx.get("floor", 0)) or 0),
        }
        if act == 1 and ctx.get("act_name"):
            out["act1"] = _norm_act(str(ctx["act_name"]))
        # The game's `RoomType` name, the mod's `current_room`. Once the map
        # opens the game names `Map`, while the run the student was trained
        # on still stands in the room it left: Neow's at the run's first map,
        # the room a `?` rolled (GH #1664).
        self.saw(state)
        if self.last_room_type:
            out["current_room"] = self.last_room_type
        if bridge_map is not None:
            out["map"] = bridge_map
        return out

    def _miss(self, why: str) -> None:
        self.unavailable[why.split(" (")[0][:80]] += 1
        logger.info("instinct unavailable: %s", why)

    # -- screens -----------------------------------------------------------

    def screen(
        self, bridge_state: dict[str, Any], state: dict[str, Any], bridge_map: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """The answer (``{"action", "index"/"indices"}``) for a screen, or None."""
        msg = {**bridge_state, **self._view(bridge_state, state, bridge_map)}
        msg["run_state"] = {**msg.get("run_state", {}), "ascension": self.ascension}
        if msg.get("type") == "event":
            msg["options"] = [_event_option(o) for o in msg.get("options") or []]
        if msg.get("type") == "card_select" and state.get("source") == "FromChooseACardScreen":
            msg = _choose_a_card(msg)
        self.event_button = None
        if msg.get("type") == "card_select" and int(msg.get("max_count") or 1) > 1:
            return self._multi_select(msg)
        best = self._ask(msg)
        if msg.get("type") == "event" and best is not None:
            pressed = next((o for o in msg["options"] if o.get("index") == best.get("index")), None)
            self.event_button = str((pressed or {}).get("text_key") or "") or None
        return best

    def _view(
        self, bridge_state: dict[str, Any], state: dict[str, Any], bridge_map: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The screen's envelope, as the engine the student was trained on stands.

        The game offers an act's boss relic at the next act's starting
        Ancient; the engine at the act's end, in the boss's room, before the
        next act opens. So that offer is asked from the view of the screen
        before it, the act's last.
        """

        def act_of(view: tuple[dict[str, Any], dict[str, Any] | None]) -> int:
            return int((view[0].get("context") or {}).get("act") or 1)

        here = self.envelope(state, bridge_map)
        if self.last_view is not None and act_of(self.last_view) < here["act"]:
            self.act_end_view = self.last_view
        self.last_view = (state, bridge_map)
        end = self.act_end_view
        if _is_boss_relic_offer(bridge_state) and end is not None:
            prev_state, prev_map = end
            if act_of(end) == here["act"] - 1:
                room, room_floor = self.last_room_type, self.last_room_floor
                try:
                    return self.envelope(prev_state, prev_map)
                finally:
                    self.last_room_type, self.last_room_floor = room, room_floor
        return here

    def potion_swap(
        self, bridge_state: dict[str, Any], state: dict[str, Any], bridge_map: dict[str, Any] | None
    ) -> int | None:
        """The belt slot to empty for a full-belt potion reward, or None to decline.

        Asked as the mod's reward screen asks it: one ``potion_swap`` option
        per answer, a decline and one per held potion.
        """
        msg = {**bridge_state, **self.envelope(state, bridge_map)}
        msg["run_state"] = {**msg.get("run_state", {}), "ascension": self.ascension}
        reward = {"id": state.get("potion")}
        msg["type"] = "reward_screen"
        msg["options"] = [
            {
                "index": int(o.get("index", i)),
                "action": "potion_swap",
                "reward_button": 0,
                "discard_slot": o.get("discard_slot"),
                "label": o.get("label"),
                "reward": reward,
            }
            for i, o in enumerate(state.get("options") or [])
        ]
        best = self._ask(msg, "potion_swap")
        if best is None:
            return None
        idx = int(best.get("index", 0))
        chosen = next((o for o in msg["options"] if o["index"] == idx), None)
        return None if chosen is None else chosen["discard_slot"]

    def _ask(self, msg: dict[str, Any], advice_for: str | None = None) -> dict[str, Any] | None:
        try:
            advice = self.adv.advise(msg, advice_for or str(msg["type"]))
        except (InstinctUnavailable, ValueError) as exc:
            self._miss(str(exc))
            if msg.get("type") == "event":
                opts = [o.get("text_key") for o in msg.get("options") or []]
                logger.info("instinct event miss: %s", opts)
            elif msg.get("type") == "map_select":
                m = msg.get("map") or {}
                logger.info(
                    "instinct map miss: act %s floor %s current %s nodes %s visited %s",
                    msg.get("act"),
                    msg.get("floor"),
                    m.get("current"),
                    [(n.get("col"), n.get("row")) for n in msg.get("nodes") or []],
                    m.get("visited"),
                )
            elif msg.get("type") == "card_select":
                logger.info(
                    "instinct card screen miss: %s %s",
                    msg.get("prompt_key"),
                    msg.get("source_model"),
                )
            return None
        best = advice.get("best")
        if not isinstance(best, dict) or advice.get("hand_back"):
            self._miss(str(advice.get("hand_back") or "no best"))
            return None
        return best

    def _multi_select(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """A pick-N deck screen in one answer.

        The mod's screen asks again after each click; the referee takes all N
        at once. So Instinct is asked once per pick, with the cards it took
        off the offer, until it confirms or N are taken. The advisor replays
        the earlier picks in its engine before each later one (a grant,
        GH #1673; enchant, remove, upgrade, transform, GH #1715).
        """
        lo, hi = int(msg.get("min_count") or 0), int(msg.get("max_count") or 1)
        cards = list(msg.get("cards") or [])
        pos = {int(c.get("index", -1)): k for k, c in enumerate(cards)}
        picked: list[int] = []
        while len(picked) < min(hi, len(cards)):
            # The whole screen and the picks so far (positions, in order):
            # the advisor replays them in the engine where it can (a grant,
            # GH #1673), else asks over the cards left.
            ask = {**msg, "picked_indices": [pos[i] for i in picked]}
            best = self._ask(ask)
            if best is None:
                return None
            if best.get("action") != "choose":
                if len(picked) >= lo:
                    break
                return None
            idx = int(best.get("index", -1))
            if idx in picked or idx not in pos:
                return None
            picked.append(idx)
        if len(picked) < lo:
            return None
        return {"action": "choose", "indices": picked} if picked else {"action": "skip"}

    # -- fights ------------------------------------------------------------

    def fight_started(self, state: dict[str, Any] | None = None) -> None:
        """A fight screen arrived with no live mirror.

        With ``state``, that is a new fight only when no fight is in progress
        on its floor: a mirror dropped in the middle of a fight leaves the
        fight's encounter and starting HP as they were (GH #1752).
        """
        floor = ((state or {}).get("context") or {}).get("total_floor")
        if state is not None and floor is not None and floor == self.fight_floor:
            return
        self.fight_floor = floor
        self.encounter = None
        self.fight_hp = None

    def enemies_renamed(self, pairs: list[tuple[int, int]] | None) -> None:
        """The fight mirror renamed enemy uids (GH #1651): the intent log follows."""
        if pairs:
            self.adv.remap_enemies(pairs)

    def _fight_msg(
        self,
        state: dict[str, Any],
        bridge_run_state: dict[str, Any],
        bridge_map: dict[str, Any] | None,
        node: str = "",
    ) -> dict[str, Any]:
        """The run behind a fight decision, as the graft reads it. ``node`` is
        the map node's room type (``unknown`` for a ``?``)."""
        # The run's HP stays at the fight's start until the fight ends, as in
        # the engine the student was trained on (RUNCTX `R_HP`, GH #1666);
        # the fight's own HP is the player creature's. Blood Vial heals at
        # the first turn's start, before the first decision (GH #1705).
        if self.fight_hp is None:
            # A choice before the first turn (a start-of-fight card pick)
            # comes before the heal.
            turn_started = state.get("decision") == "combat_play"
            room = str((state.get("context") or {}).get("room_type") or "")
            self.fight_hp = _fight_start_hp(
                bridge_run_state,
                self.room_hp,
                healed=turn_started,
                boss=room.lower() == "boss",
                unknown=node.lower() == "unknown",
                from_event=bool(self.event_button),
            )
        self.event_button = None
        return {
            "type": "combat_action",
            "run_state": {**bridge_run_state, **self.fight_hp, "ascension": self.ascension},
            **self.envelope(state, bridge_map),
        }

    def _name_encounter(self, state: dict[str, Any], kinds: list[str], room_type: str) -> None:
        """Name the fight's encounter on its first decision (a turn or a screen)."""
        if self.encounter is not None:
            return
        ctx = state.get("context") or {}
        # The host names it (`encounter_id`, PROTOCOL question 8); an older
        # view does not, and then it is read off the roster. Whether an event
        # started the fight is the event's either way: the student reads such
        # a fight's room as `event` (GH #1937: once the id was sent, this was
        # skipped and every event fight tokenized as a `monster` room).
        named = state.get("encounter_id")
        if self.event_button:
            from_event = event_encounter(self.event_button, self.game_version)
            self.event_fight = from_event is not None
            if not (isinstance(named, str) and named):
                named = from_event
        # `event_button` is cleared by `_fight_msg`, which still reads it for
        # the fight's starting HP (it runs after this, GH #1751).
        self.encounter = (
            named
            if isinstance(named, str) and named
            else infer_encounter(
                str(ctx.get("act_name") or ""), _fight_tier(state, room_type), kinds
            )
            or ""
        )
        if not self.encounter:
            logger.info(
                "instinct: no encounter for %s %s %s", ctx.get("act_name"), room_type, kinds
            )

    def combat_choice(
        self,
        state: dict[str, Any],
        cs: Any,
        max_select: int,
        *,
        room_type: str,
        bridge_run_state: dict[str, Any],
        bridge_map: dict[str, Any] | None,
    ) -> list[int] | None:
        """The student's option picks on an in-fight choice, or None.

        ``cs`` is the fight mirror's state parked on the choice. Natively the
        student answers it (COMBAT_CHOICE), not the search (GH #1667).
        """
        # A fight that opens on a screen is named here (GH #1697).
        fight = sts2_rs.state_to_dict(cs) if cs is not None else {}
        kinds = [e.get("kind") for e in fight.get("encounter") or []]
        board = state if state.get("context") else (state.get("combat") or state)
        self._name_encounter(board, [str(k) for k in kinds], room_type)
        msg = self._fight_msg(state, bridge_run_state, bridge_map, node=room_type)
        try:
            rm, _notes = sts2_rs.graft_run(msg, self.game_version)
            room = _fight_tier(state, room_type)
            sts2_rs.instinct_enter_live_combat(rm, cs, {"encounter_id": self.encounter or ""}, room)
            return self.adv.choice_picks(msg, rm, max_select)
        except (
            InstinctUnavailable,
            sts2_rs.GraftUnavailableError,
            ValueError,
            RuntimeError,
        ) as exc:
            self._miss(str(exc))
            return None

    def combat(
        self,
        state: dict[str, Any],
        wire: dict[str, Any],
        cs: Any,
        room_type: str,
        bridge_run_state: dict[str, Any],
        bridge_map: dict[str, Any] | None,
    ) -> list[dict[str, Any]] | None:
        """The student's engine actions at a fight decision, best first, or None.

        ``cs`` is the fight as the combat mirror holds it; ``wire`` the board
        it was reconciled from. The actions are in the engine's own dialect
        (``play`` with an engine hand index, ``potion`` with a slot,
        ``end_turn``), as ``CombatMirror.advance`` takes them; the caller
        plays the first one the game allows.
        """
        # The fight's opening roster: the board, with the enemies that died
        # and left it since (GH #1716), on the first decision already.
        opened = wire.get("removed_roster") or wire.get("encounter") or []
        self._name_encounter(state, [e["kind"] for e in opened], room_type)
        msg = self._fight_msg(state, bridge_run_state, bridge_map, node=room_type)
        try:
            rm, _notes = sts2_rs.graft_run(msg, self.game_version)
            room = _fight_tier(state, room_type)
            sts2_rs.instinct_enter_live_combat(rm, cs, {"encounter_id": self.encounter}, room)
            pick = self.adv.evaluate(msg, rm, ("COMBAT_TURN",))
        except (InstinctUnavailable, sts2_rs.GraftUnavailableError, ValueError) as exc:
            self._miss(str(exc))
            return None
        classes = pick.decision.get("classes") or []
        order = sorted(range(len(pick.probs)), key=lambda c: -pick.probs[c])
        by_class = {int(c.get("class", i)): c for i, c in enumerate(classes)}
        ranked = [
            a
            for ci in order
            for a in (by_class.get(ci) or {}).get("actions") or []
            if isinstance(a, dict)
        ]
        if not ranked:
            self._miss("no action in the student's classes")
            return None
        return ranked
