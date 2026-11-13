"""Replay a recorded real-game run into the headless host and report where they part.

A SpireCodex replay (`<hash>.replay.ndjson.gz`) logs a run played in the real
game: the seed, every decision with the options shown and the one taken, and
in combat every draw, play, enemy move and hit. This module starts the host on
the same seed, sends it the human's choices one by one, and compares what the
host produces with what the game produced. The first disagreement ends the
replay; its kind and detail are the result.

    python python/host_replay.py --run <file>            # one run, verbose
    python python/host_replay.py --census 300 --out DIR  # sample and tally

GH #1286. The log holds events, not full state: a match means everything the
log records agrees, not that every hidden field does.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from sts2_client import Game, set_game_version

GAME_VERSION = "v0.111.0"

#: The coach mod under every name it shipped as: its runs are not human play.
COACH_MOD_IDS = frozenset({"SpireMentor", "SpireCoach", "SimBridge"})

#: Replay stream names -> host `rng_counters` names.
_STREAM = {
    "player_rewards": "rewards",
    "player_shops": "shops",
    "player_transformations": "transformations",
}

_DECK_SELECT = "deck_select_"

#: Runs the player left before this floor say too little to be worth a host start.
MIN_FLOORS = 4

#: GH #1282: the host lists a reward screen as a `rewards` decision (take or skip
#: each reward, in any order) instead of taking the rewards itself. Same switch
#: the host reads.
INTERACTIVE_REWARDS = os.environ.get("STS2_INTERACTIVE_REWARDS") == "1"

#: Log `resolve` reward kinds -> host `rewards` entry types.
_REWARD_TYPE = {
    "GoldReward": "gold",
    "PotionReward": "potion",
    "RelicReward": "relic",
    "CardReward": "card",
    "SpecialCardReward": "special_card",
    "CardRemovalReward": "card_removal",
}


class Desync(Exception):
    """The host and the log disagree."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


class LogEnd(Exception):
    """The log has no more choices: the replay is over, in sync."""


def replays_root() -> str:
    env = os.environ.get("STS2_RUN_SCRAPER")
    if env:
        return str(Path(env) / "data" / "replays-spire-codex")
    for parent in Path(__file__).resolve().parents:
        cand = parent.parent / "sts2_run_scraper"
        if cand.is_dir():
            return str(cand / "data" / "replays-spire-codex")
    return "/ssdpool/thomas/projects/sts2/sts2_run_scraper/data/replays-spire-codex"


def load(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # truncated upload: keep what parsed
    return rows


def _neow_twice(rows: list[dict[str, Any]]) -> bool:
    """The run's first event was opened twice: an unflagged restart."""
    return (
        sum(
            1
            for r in rows[:60]
            if r["t"] == "decision" and r.get("event_id") == "NEOW" and r.get("page_index") == 0
        )
        > 1
    )


def eligible(rows: list[dict[str, Any]]) -> str | None:
    """None if the run can be replayed, else the reason it cannot."""
    if not rows or rows[0].get("t") != "header":
        return "no_header"
    h = rows[0]
    if h.get("build_id") != GAME_VERSION:
        return "version"
    if (h.get("replay_version") or 0) < 7 or not any(r["t"] == "room_exit" for r in rows):
        return "replay_version"  # older logs have no room boundaries
    if max((r.get("floor") or 0) for r in rows) < MIN_FLOORS:
        return "short"
    if (h.get("player_count") or 1) != 1 or h.get("game_mode") != "standard" or h.get("modifiers"):
        return "mode"
    if any(m.get("affects_gameplay") or m.get("id") in COACH_MOD_IDS for m in h.get("mods") or []):
        return "mods"
    if h.get("reloads") or any(r["t"] in ("resume", "room_resume") for r in rows):
        return "reload"
    if _neow_twice(rows) or any(r["t"] == "remap" for r in rows):
        return "reload"  # the room was loaded again without a `resume` row
    if any(r["t"] == "console" for r in rows):
        return "console"
    open_fight = False
    for r in rows:
        if r["t"] == "combat_start":
            if open_fight:
                return "reload"  # a fight restarted without a `resume` row
            open_fight = True
        elif r["t"] == "combat_end":
            open_fight = False
    return None


def _norm(card_id: str) -> str:
    return card_id.removeprefix("CARD.").removesuffix("_CARD")


def _loose(key: Any) -> str:
    """An event option id for comparison: some log a title where the host has `<ID>.title`."""
    return "".join(ch for ch in str(key).removesuffix(".title").lower() if ch.isalnum())


def _card(c: dict[str, Any]) -> tuple[str, bool]:
    """(id, upgraded) of a host card or a log card/option."""
    cid = c.get("id") or c.get("option_id") or ""
    return _norm(cid), bool(c.get("upgraded") or c.get("up"))


def _is_combat_choice(r: dict[str, Any]) -> bool:
    t = r["t"]
    return (
        (t == "play" and not r.get("auto") and not r.get("play_index"))
        or t in ("potion_start", "potion_used", "potion_dropped")
        or (t == "end_turn" and r.get("side") == "player")
        or (t == "pick" and r.get("selector") == "human")
    )


def _with_missing_end_turns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add the player `end_turn` rows the recorder left out.

    Some enemy turns start with no player `end_turn` row before them, though the
    player's turn did end. Without the row the replay would never end that turn.
    """
    out: list[dict[str, Any]] = []
    ended = True
    for r in rows:
        t, side = r["t"], r.get("side")
        if t in ("combat_start", "combat_end"):
            ended = True
        elif (t == "turn" and side == "player") or (t == "energy" and r.get("src") == "turn"):
            # A player turn starts with its energy refill; some logs have no
            # `turn` row for it.
            ended = False
        elif t == "end_turn" and side == "player":
            ended = True
        elif t == "turn" and side == "enemy" and not ended:
            out.append({"t": "end_turn", "side": "player", "floor": r.get("floor"), "added": True})
            ended = True
        out.append(r)
    return out


def _hits_lost(fight: list[dict[str, Any]]) -> bool:
    """Whether an enemy's attack move in this fight has no `hit` row at all."""
    waiting = False
    for r in fight:
        t = r["t"]
        if t == "hit" or t == "death":
            waiting = False
        elif t in ("move", "end_turn", "combat_end"):
            if waiting:
                return True
            waiting = t == "move" and "attack" in (r.get("intents") or [])
    return False


def _foe_hits_lost_from(fight: list[dict[str, Any]]) -> int | None:
    """Offset in `fight` after which the log misses hits on an enemy, if it does.

    An enemy that dies to an attack has a `hit` row with `killed`. When that row is
    missing, the hits since the last one logged on that enemy were dropped.
    """
    first: int | None = None
    last_hit: dict[Any, int] = {}
    killed: set[Any] = set()
    for i, r in enumerate(fight):
        t = r["t"]
        if t == "hit" and r.get("dst_cid") is not None:
            last_hit[r["dst_cid"]] = i
            if r.get("killed"):
                killed.add(r["dst_cid"])
        elif (
            t == "death"
            and r.get("killed_by") == "ATTACK_COMMAND"
            and not r.get("removal_prevented")
            and r.get("tgt_cid") is not None
            and r["tgt_cid"] not in killed
        ):
            at = last_hit.get(r["tgt_cid"], 0)
            first = at if first is None else min(first, at)
    return first


class Tracker:
    """What the log says the player and the enemies look like, fed row by row."""

    def __init__(self, header: dict[str, Any]) -> None:
        self.character = header.get("character")
        self.max_hp: int = header.get("starting_max_hp", 0)
        self.hp: int = 0  # the log opens with a heal to the starting hp
        self.gold: int | None = None
        self.deck: list[dict[str, Any]] = list(header.get("starting_deck", []))
        self.relics: list[str] = list(header.get("starting_relics", []))
        self.belt: list[str] = []
        self.enemies: dict[int, list[Any]] = {}  # cid -> [monster id, hp]
        self.hp_events_carry_damage = False
        self.prev: dict[str, Any] = {}
        self.hp_unknown = False  # the recorder dropped a hit on the player
        self.hp_seen = False  # no hp reading yet (a run at A0 starts with none)
        self.start_hp = 0
        self.heals = 0
        self._attack: list[int] | None = None  # [enemy cid, hits announced, hits seen]
        # The hand in the order it was drawn, as [instance, card id]. A generated card
        # is listed although it may have gone to another pile (see `hand_rank`).
        self.hand: list[list[Any]] = []
        self.draw_pile: list[Any] = []  # instances in draw order, from the shuffle rows
        self.ids: dict[Any, str] = {}  # instance -> card id

    def hand_rank(self, c: Any, n_host: int) -> int | None:
        """Position of instance `c` among the copies of its card in its pile.

        The pile is the hand, else the draw pile. None when the host shows more
        copies than the log's pile holds: then the order is unknown.
        """
        mine = next((x for x in self.hand if x[0] == c), None)
        if mine is not None:
            same = [x[0] for x in self.hand if x[1] == mine[1]]
        elif c in self.draw_pile and c in self.ids:
            same = [x for x in self.draw_pile if self.ids.get(x) == self.ids[c]]
        else:
            return None
        # The host can offer fewer copies than the pile holds (a pick that leaves out
        # cards already retained): those left out are the older ones, so count from
        # the end.
        at = same.index(c) - (len(same) - n_host)
        return at if 0 <= at < n_host else None

    def _move_card(self, d: dict[str, Any]) -> None:
        t = d["t"]
        if d.get("c") is not None and d.get("id"):
            self.ids[d["c"]] = _norm(str(d["id"]))
        if t in ("draw", "play", "discard", "exhaust") and d.get("c") in self.draw_pile:
            self.draw_pile.remove(d["c"])
        if t == "draw_order":
            self.draw_pile = list(d.get("order_c") or [])
            by_deck = {c.get("c"): c.get("id") for c in self.deck}
            for c, dc in zip(self.draw_pile, d.get("order_deck_c") or [], strict=False):
                if dc in by_deck:
                    self.ids[c] = _norm(str(by_deck[dc]))
        elif t == "shuffle":
            self.draw_pile = list(d.get("order_c") or [])
            # A generated card listed here went to a pile, not to the hand.
            self.hand = [x for x in self.hand if x[0] not in self.draw_pile]
        if t == "combat_start":
            self.hand = []
            self.draw_pile = []
        elif t in ("draw", "generate"):
            self.hand = [x for x in self.hand if x[0] != d.get("c")]
            if t == "draw" or d.get("c") not in self.draw_pile:  # not a transform's echo
                self.hand.append([d.get("c"), _norm(str(d.get("id"))), t == "draw"])
        elif t in ("play", "discard", "exhaust"):
            self.hand = [x for x in self.hand if x[0] != d.get("c")]
        elif t == "flush":
            gone = set(d.get("flushed_c") or [])
            self.hand = [x for x in self.hand if x[0] not in gone]
        elif t == "transform":
            for x in self.hand:
                if x[0] == d.get("from_c"):
                    x[0], x[1] = d.get("to_c"), _norm(str(d.get("to_id")))
            self.draw_pile = [d.get("to_c") if x == d.get("from_c") else x for x in self.draw_pile]
            self.ids[d.get("to_c")] = _norm(str(d.get("to_id")))

    def _close_attack(self) -> None:
        if self._attack is not None and self._attack[2] < self._attack[1]:
            self.hp_unknown = True
        self._attack = None

    def feed(self, d: dict[str, Any]) -> None:
        self._feed(d)
        self.prev = d

    def _feed(self, d: dict[str, Any]) -> None:
        t = d["t"]
        self._move_card(d)
        if t in ("attack", "move", "end_turn", "combat_end"):
            self._close_attack()
            if t == "attack" and d.get("src_cid") in self.enemies and not d.get("card"):
                self._attack = [d["src_cid"], int(d.get("hits") or 0), 0]
        elif t == "hit" and self._attack is not None and d.get("src_cid") == self._attack[0]:
            self._attack[2] += 1
        elif t == "death" and self._attack is not None and d.get("tgt_cid") == self._attack[0]:
            self._attack = None  # the attacker died mid-attack (thorns)
        if t == "combat_start":
            self.start_hp, self.heals, self.hp_unknown = self.hp, 0, False
        elif t == "hp" and d.get("dst", "player") == "player" and d.get("d", 0) > 0:
            self.heals += d["d"]
        elif t == "combat_end" and self.hp_unknown and d.get("hp_lost_total") is not None:
            # Hits were dropped in this fight: fall back on its logged total.
            self.hp = self.start_hp - d["hp_lost_total"] + self.heals
            self.hp_unknown = False
        if t == "deck":
            self.deck = d["cards"]
        elif t in ("gold", "shop"):
            self.gold = d["gold"]
        elif t == "buy" and "gold_on_hand" in d:
            self.gold = d["gold_on_hand"]  # a purchase logs no `gold` row
        elif t == "relic" and d.get("id"):
            self.relics.append(d["id"])
        elif t == "relic_lost" and d.get("id") in self.relics:
            self.relics.remove(d["id"])
        elif t == "potion_got":
            self.belt.append(d["id"])
        elif t in ("potion_used", "potion_dropped"):
            if d["id"] in self.belt:
                self.belt.remove(d["id"])
        elif t == "max_hp":
            if d.get("dst", "player") == "player":
                self.max_hp = d.get("max_hp", self.max_hp)
                self.hp = d.get("hp", self.hp)
            elif d.get("dst_cid") in self.enemies and "hp" in d:
                self.enemies[d["dst_cid"]][1] = d["hp"]
        elif t == "hp":
            if d.get("dst", "player") == "player":
                # The row carries the hp after the change: resync on it, so a hit the
                # recorder dropped earlier does not skew every later comparison.
                self.hp = d["hp"] if isinstance(d.get("hp"), int) else self.hp + d.get("d", 0)
                self.hp_unknown = False
                self.hp_seen = True
            elif d.get("dst_cid") in self.enemies and d.get("d", 0) > 0:
                self.enemies[d["dst_cid"]][1] += d["d"]
        elif t == "hit":
            if d.get("dst") == "player":
                if not self.hp_events_carry_damage:
                    self.hp -= d.get("dmg", 0)
            elif d.get("dst_cid") in self.enemies:
                self.enemies[d["dst_cid"]][1] -= d.get("dmg", 0)
        elif t == "hp_loss":
            # Often the echo of the `hp` row just before it: count the loss once.
            p = self.prev
            if not (
                p.get("t") == "hp"
                and p.get("dst", "player") == "player"
                and p.get("d") == -d.get("dmg", 0)
            ):
                self.hp -= d.get("dmg", 0)
        elif t == "spawn" and d.get("side") == "enemy":
            self.enemies[d["cid"]] = [d["id"], d["hp"]]
        elif t == "death" and d.get("tgt_cid") in self.enemies:
            if not d.get("removal_prevented"):
                self.enemies[d["tgt_cid"]][1] = 0
        elif t == "escape" and d.get("tgt_cid") in self.enemies:
            self.enemies[d["tgt_cid"]][1] = 0


class Replayer:
    def __init__(self, path: str, *, verbose: bool = False) -> None:
        self.path = path
        self.verbose = verbose
        self.rows = _with_missing_end_turns(load(path))
        self.header = self.rows[0]
        self.track = Tracker(self.header)
        self.fed = 1  # rows[:fed] have been fed to the tracker
        self.used: set[int] = set()
        # A room's rows end at its depth-1 `room_exit`; the next room's rows
        # (its decisions, then its `room` row) follow.
        self.segs: list[tuple[int, int]] = []
        start = 1
        for i, r in enumerate(self.rows):
            if r["t"] == "room_exit" and r.get("depth", 1) == 1:
                self.segs.append((start, i + 1))
                start = i + 1
        self.segs.append((start, len(self.rows)))
        self.k = 0
        self.in_combat = False
        self.ci = 0  # combat cursor: next row to look at
        self.combat_end = 0  # index of the current combat's `combat_end` (or log end)
        self.next_choice: int | None = None
        self.pending_potion: Counter[str] = Counter()
        self.hits_lost = False  # this fight's log has an attack with no `hit` rows
        self.foe_blind_from: int | None = None  # row after which enemy hp is unknown
        self.hand_noted = False  # this fight's first hand difference is already noted
        self.actions = 0
        self.combats = 0
        self.notes: list[str] = []
        self.last_rest: str | None = None
        self.removed = {r.get("c") for r in self.rows if r["t"] == "remove"}
        self.dec_rows = {
            r["decision_id"]: i for i, r in enumerate(self.rows) if r["t"] == "decision"
        }
        self.g: Game | None = None
        # Reward screen (interactive rewards): the room whose card rewards are
        # mapped, and its card-reward decisions not yet taken, in log order.
        self.screen_k: int | None = None  # first row of the mapped screen
        self.screen_cards: list[int] = []
        self.rerolled: set[int] = set()  # card-reward decisions already rerolled

    # ---- log helpers -------------------------------------------------

    def log(self, *a: Any) -> None:
        if self.verbose:
            print(*a, file=sys.stderr)

    def note(self, s: str) -> None:
        if len(self.notes) < 20:
            self.notes.append(s)

    def feed_to(self, idx: int) -> None:
        for i in range(self.fed, min(idx, len(self.rows))):
            before = (self.track.hp, self.track.gold)
            self.track.feed(self.rows[i])
            if self.verbose and before != (self.track.hp, self.track.gold):
                self.log(
                    f"     log hp/gold {before} -> {(self.track.hp, self.track.gold)} by {json.dumps(self.rows[i])[:200]}"
                )
        self.fed = max(self.fed, idx)

    def seg(self) -> range:
        a, b = self.segs[self.k]
        return range(a, b)

    def in_some_combat(self, i: int) -> bool:
        """Whether row `i` lies between a `combat_start` and its `combat_end`."""
        for j in range(i, self.segs[self.k][0] - 1, -1):
            t = self.rows[j]["t"]
            if t == "combat_end":
                return j == i
            if t == "combat_start":
                return True
        return False

    def resolution(self, dec: dict[str, Any]) -> dict[str, Any] | None:
        """The row that says what the human did with decision `dec`."""
        did, dtype = dec["decision_id"], dec.get("decision_type", "")
        i = self.dec_rows[did] + 1
        # The `outcome` of a deck pick can be logged after the room's exit.
        end = min(len(self.rows), self.segs[self.k][1] + 40)
        found: dict[str, Any] | None = None
        inst = {o.get("instance_id") for o in dec.get("options", [])} - {None}
        for r in self.rows[i:end]:
            if r.get("decision_id") != did:
                continue
            t = r["t"]
            if dtype.startswith(_DECK_SELECT) and t == "pick":
                return r
            if t == "outcome":
                return r
            if (
                dtype == "card_reward"
                and t == "acquire"
                and r.get("option_index") is not None
                and r.get("c") not in self.removed  # not a stolen card coming back
            ):
                return r
            if dtype == "rest" and t == "rest":
                return r
            if dtype in ("relic_reward", "treasure") and t in ("resolve", "relic"):
                found = found or r
        if found is None and dtype == "card_reward":
            # With several rewards on one screen the `acquire` can carry no decision
            # id, or another reward's: go by the card instance.
            for r in self.rows[i:end]:
                if r["t"] == "acquire" and r.get("c") in inst:
                    return r
        return found

    def pending(self) -> list[int]:
        """Unconsumed rows of this room that record a human choice, in log order."""
        out = []
        combat_open = False
        for i in self.seg():
            r = self.rows[i]
            t = r["t"]
            if t == "combat_start":
                combat_open = True
                if i not in self.used:
                    out.append(i)
            elif t == "combat_end":
                combat_open = False
            elif combat_open or i in self.used:
                continue
            elif t == "decision":
                dtype = r.get("decision_type", "")
                if dtype == "map_node" or not r.get("options"):
                    continue
                res = self.resolution(r)
                if dtype in ("relic_reward", "treasure"):
                    continue  # the host takes these itself; checked in leave_room
                if res is None or res.get("outcome") in ("skip", "decline"):
                    continue
                if dtype.startswith(_DECK_SELECT) and len(r["options"]) <= (
                    r.get("min_select") or 0
                ):
                    continue  # nothing to choose: the host resolves it without asking
                out.append(i)
            elif t in ("buy", "potion_used", "potion_dropped") or (
                t == "pick"
                and r.get("selector") == "human"
                and not r.get("decision_id")
                and r.get("cards")
            ):
                out.append(i)
        return out

    # ---- host helpers ------------------------------------------------

    def act(self, action: str, **args: Any) -> dict[str, Any]:
        assert self.g is not None
        self.log("  ->", action, args)
        d = self.g.act(action, **args)
        self.actions += 1
        if d.get("type") == "error":
            raise Desync("host_rejected", f"{action} {args}: {str(d.get('message'))[:200]}")
        if d.get("type") != "decision":
            raise Desync("host_reply", json.dumps(d)[:200])
        if self.verbose and d.get("decision") == "combat_play":
            st = self.g.get_state()
            pl = st["player"]
            foes = [
                (
                    e.get("hp"),
                    e.get("block"),
                    [(p.get("id"), p.get("amount")) for p in e.get("powers") or []],
                )
                for e in st.get("enemies") or d.get("enemies") or []
            ]
            self.log(f"     host hp {pl.get('hp')} block {pl.get('block')} foes {foes}")
            self.log(f"     host hand {[c.get('id') for c in d.get('hand', [])]}")
            self.log(f"     log  hand {[x[1] for x in self.track.hand]} (fed {self.fed})")
        return d

    # ---- combat ------------------------------------------------------

    def scan_choice(self) -> None:
        """Set `next_choice` to the next human combat row, or None at combat end."""
        i = self.ci
        while i < self.combat_end:
            r = self.rows[i]
            if _is_combat_choice(r):
                if r["t"] == "potion_used" and self.pending_potion[str(r.get("id"))] > 0:
                    self.pending_potion[r["id"]] -= 1  # the echo of a `potion_start`
                else:
                    self.next_choice = i
                    self.feed_to(i)
                    return
            i += 1
        self.next_choice = None
        self.feed_to(self.combat_end + 1)

    def enter_combat(self, d: dict[str, Any]) -> None:
        start = next(
            (i for i in self.seg() if i not in self.used and self.rows[i]["t"] == "combat_start"),
            None,
        )
        if start is None:
            pend = self.pending()
            raise Desync(
                "extra_combat",
                "host is in a fight the log does not have here; log next: "
                + (json.dumps(self.rows[pend[0]])[:200] if pend else "nothing"),
            )
        earlier = [i for i in self.pending() if i < start]
        if earlier:
            raise Desync("unconsumed", "before combat: " + json.dumps(self.rows[earlier[0]])[:300])
        row = self.rows[start]
        self.used.add(start)
        end = next(
            (i for i in range(start, len(self.rows)) if self.rows[i]["t"] == "combat_end"),
            len(self.rows),
        )
        if any(self.rows[i]["t"] == "combat_start" for i in range(start + 1, end)):
            raise Desync("log_reload", "second combat_start before combat_end")
        self.combat_end = end
        self.feed_to(start)
        self.track.enemies = {
            e["cid"]: [e["id"], e["hp"]] for e in row.get("enemies", []) if "cid" in e
        }
        self.track.hp_events_carry_damage = any(
            r["t"] == "hp" and r.get("d", 0) < 0 and r.get("dst", "player") == "player"
            for r in self.rows[start:end]
        )
        self.hand_noted = False
        self.hits_lost = _hits_lost(self.rows[start:end])
        off = _foe_hits_lost_from(self.rows[start:end])
        self.foe_blind_from = None if off is None else start + off
        if off is not None:
            self.note(f"foe_hits_lost f{row.get('floor')} {row.get('encounter')}")
        if self.hits_lost:
            self.note(f"hits_lost f{row.get('floor')} {row.get('encounter')}")
        self.in_combat = True
        self.ci = start + 1
        self.pending_potion.clear()
        self.log(f"== combat {row.get('encounter')} floor {row.get('floor')}")

        assert self.g is not None
        st = self.g.get_state()
        rng = row.get("rng_state") or {}
        hc = st.get("rng_counters") or {}
        diff = {
            k: (hc[_STREAM.get(k, k)], v)
            for k, v in rng.items()
            if _STREAM.get(k, k) in hc and hc[_STREAM.get(k, k)] != v
        }
        if diff:
            self.note(f"rng@{row.get('encounter')} f{row.get('floor')} host/log {diff}")

        self.scan_choice()
        # Start-of-fight effects (a relic's opening damage) are logged after
        # `combat_start`, so compare once the log has reached the first choice.
        host = Counter((m["kind"], m["hp"]) for m in st.get("monsters", []) if m.get("alive"))
        want = Counter((mid, hp) for mid, hp in self.track.enemies.values() if hp > 0)
        if self.next_choice is not None and self.rows[self.next_choice]["t"] == "end_turn":
            # A first turn with no choice before `end_turn` (every card played
            # by itself): the log has the end-of-turn effects before that row,
            # the host has not run them yet. Hp is compared after the turn.
            host = Counter(kind for kind, _ in host.elements())
            want = Counter(kind for kind, _ in want.elements())
        if host != want:
            raise Desync("encounter", f"host {sorted(host.items())} log {sorted(want.items())}")
        if not self.track.hp_seen:
            self.track.hp, self.track.hp_seen = st["player"]["hp"], True
        if st["player"]["hp"] != self.track.hp and not self.track.hp_unknown:
            raise Desync("hp_precombat", f"host {st['player']['hp']} log {self.track.hp}")
        first = self.next_choice if self.next_choice is not None else end
        opening = Counter(_norm(r["id"]) for r in self.rows[start:first] if r["t"] == "draw")
        gone = Counter(
            _norm(r["id"])
            for r in self.rows[start:first]
            if r["t"] in ("discard", "exhaust") or (r["t"] == "play" and r.get("auto"))
        )
        hand = Counter(_norm(c["id"]) for c in (d.get("combat") or d).get("hand", []))
        # The log does not record every card that starts in hand (generated
        # and innate ones), so only a logged draw the host lacks counts.
        if opening and not gone and opening - hand:
            raise Desync(
                "opening_hand",
                f"host {sorted(hand.elements())} log {sorted(opening.elements())}",
            )

    def check_combat(self) -> None:
        assert self.g is not None
        st = self.g.get_state()
        if self.hits_lost:
            return  # neither side's hp can be rebuilt from this fight's log
        if st["player"]["hp"] != self.track.hp and not self.track.hp_unknown:
            raise Desync("player_hp", f"host {st['player']['hp']} log {self.track.hp}")
        if self.foe_blind_from is not None and self.fed > self.foe_blind_from + 1:
            return
        host = Counter((m["kind"], m["hp"]) for m in st.get("monsters", []) if m.get("alive"))
        want = Counter((mid, hp) for mid, hp in self.track.enemies.values() if hp > 0)
        if host != want:
            raise Desync("enemy_hp", f"host {sorted(host.items())} log {sorted(want.items())}")

    def target_index(self, d: dict[str, Any], row: dict[str, Any]) -> int | None:
        cid, name = row.get("target_cid"), row.get("target")
        if (cid is None and name is None) or name == self.track.character:
            return None
        if cid is not None and cid in self.track.enemies:
            name = self.track.enemies[cid][0]
        assert self.g is not None
        alive = [m for m in self.g.get_state().get("monsters", []) if m.get("alive")]
        shown = (d.get("combat") or d).get("enemies") or []
        if len(alive) != len(shown):
            raise Desync("target", f"{len(alive)} alive monsters, {len(shown)} shown")
        same = [i for i, m in enumerate(alive) if m["kind"] == name]
        if not same:
            raise Desync("target", f"no enemy {name} in host {[m['kind'] for m in alive]}")
        pos = same[0]
        if len(same) > 1 and cid is not None:
            cids = sorted(
                c
                for c, (mid, hp) in self.track.enemies.items()
                if mid == name and (hp > 0 or c == cid)
            )
            rank = cids.index(cid) if cid in cids else 0
            pos = same[min(rank, len(same) - 1)]
            # A summoned enemy can stand anywhere in the host's line, so the order of
            # ids is only a fallback: go by hp when the log knows it.
            hp = self.track.enemies.get(cid, [None, None])[1]
            if not self.hits_lost and hp is not None:
                by_hp = [i for i in same if alive[i]["hp"] == hp]
                twins = [c for c in cids if self.track.enemies[c][1] == hp]
                if by_hp:
                    pos = by_hp[min(twins.index(cid) if cid in twins else 0, len(by_hp) - 1)]
        return shown[pos]["index"]

    def combat_action(self, d: dict[str, Any]) -> dict[str, Any]:
        if self.next_choice is None:
            if self.combat_end >= len(self.rows):
                raise LogEnd  # the player left mid-fight
            if any(
                r["t"] == "death"
                and r.get("tgt") == "player"
                and r.get("killed_by") == "RUN_MANAGER"
                for r in self.rows[self.ci - 1 : self.combat_end]
            ):
                raise LogEnd  # the player gave the run up mid-fight
            # The fight ended while the turn was ending (an orb, poison, a
            # power): the log then has no `end_turn` row for that last turn.
            out = self.act("end_turn")
            if out["decision"] == "combat_play" or (
                out["decision"] == "card_select" and out.get("combat")
            ):
                raise Desync(
                    "combat_not_over", "log's fight ended, host's goes on after the turn's end"
                )
            return self.finish_combat(out)
        row = self.rows[self.next_choice]
        t = row["t"]
        selecting = d["decision"] == "card_select"
        self.log(f"  log {t} {row.get('id') or row.get('cards') or ''}")
        if t == "pick" and not selecting:
            # A choice with no real alternative: the host resolves it without asking.
            self.note(
                f"auto_pick f{row.get('floor')} {[c.get('id') for c in row.get('cards', [])]}"
            )
            self.ci = self.next_choice + 1
            self.scan_choice()
            return self.combat_action(d)
        if selecting != (t == "pick"):
            raise Desync("pick", f"host at {d['decision']}, log has {t} {row.get('id', '')}")
        if t == "end_turn":
            out = self.act("end_turn")
        elif t == "play":
            hand = d.get("hand", [])
            want = _card(row)
            same = [c for c in hand if _card(c) == want]
            if not same:
                raise Desync("hand", f"{want} not in host hand {[_card(c) for c in hand]}")
            playable = [c for c in same if c.get("can_play")]
            if not playable:
                raise Desync(
                    "unplayable", f"{want} in hand but not playable (energy {d.get('energy')})"
                )
            paid = row.get("cost_paid")
            src = next((c for c in self.track.deck if c.get("c") == row.get("deck_c")), None)
            ench = (src or {}).get("enchantment")
            if src is not None:
                # The host names the enchantment by its title, the log by its id.
                twin = [
                    c for c in playable if _loose(c.get("enchantment") or "") == _loose(ench or "")
                ]
                playable = twin or playable
            card = next((c for c in playable if c.get("cost") == paid), playable[0])
            # Which of several identical copies: the one at the same place in hand.
            copies = [c for c in hand if _norm(c["id"]) == want[0]]
            rank = self.track.hand_rank(row.get("c"), len(copies))
            if rank is not None and len(copies) > 1:
                at = copies[rank]
                if at in playable:
                    card = at
                else:
                    self.note(f"copy_rank_miss f{row.get('floor')} {want[0]}")
            if ench or any(c.get("enchantment") for c in same):
                self.log(
                    f"     log enchant {ench}; host copies {[c.get('enchantment') for c in same]}"
                )
            out = self.act(
                "play_card", card_index=card["index"], target_index=self.target_index(d, row)
            )
        elif t == "potion_dropped":
            belt = d.get("player", {}).get("potions") or []
            slot = next((p for p in belt if p.get("entry") == row["id"]), None)
            if slot is None:
                raise Desync(
                    "potion", f"{row['id']} not on host belt {[p.get('entry') for p in belt]}"
                )
            out = self.act("discard_potion", potion_index=slot["index"])
        elif t in ("potion_start", "potion_used"):
            if t == "potion_start":
                self.pending_potion[row["id"]] += 1
            belt = d.get("player", {}).get("potions") or []
            slot = next((p for p in belt if p.get("entry") == row["id"]), None)
            if slot is None:
                raise Desync(
                    "potion", f"{row['id']} not on host belt {[p.get('entry') for p in belt]}"
                )
            out = self.act(
                "use_potion", potion_index=slot["index"], target_index=self.target_index(d, row)
            )
        else:  # pick
            cards = d.get("cards", [])
            sel: list[int] = []
            for c in row.get("cards", []):
                want = _card(c)
                hit = next((x for x in cards if _card(x) == want and x["index"] not in sel), None)
                copies = [x for x in cards if _norm(x["id"]) == want[0]]
                rank = self.track.hand_rank(c.get("c"), len(copies))
                self.log(
                    f"     pick {c.get('c')} rank {rank} of {len(copies)} host copies; host list "
                    f"{[_norm(x['id']) for x in cards]}; log draw pile "
                    f"{[(x, self.track.ids.get(x)) for x in self.track.draw_pile]}"
                )
                if rank is not None and len(copies) > 1:
                    at = copies[rank]
                    if _card(at) == want and at["index"] not in sel:
                        hit = at
                if hit is None:
                    raise Desync(
                        "pick", f"{want} not among host options {[_card(x) for x in cards]}"
                    )
                sel.append(hit["index"])
            out = (
                self.act("select_cards", indices=",".join(map(str, sel)))
                if sel
                else self.act("skip_select")
            )

        self.ci = self.next_choice + 1
        self.scan_choice()
        if (
            t != "end_turn"
            and self.next_choice is not None
            and self.rows[self.next_choice]["t"] == "end_turn"
            and out["decision"] == "combat_play"
            and d.get("round") is not None
            and out.get("round", d["round"]) > d["round"]
        ):
            # The action ended the turn itself (Void Form); the log's `end_turn`
            # row is that same turn end, not a second one.
            self.ci = self.next_choice + 1
            self.scan_choice()
        elif (
            t == "play"
            and self.next_choice is not None
            and self.rows[self.next_choice]["t"] == "end_turn"
            and out["decision"] == "card_select"
        ):
            # Same, with the enemy turn opening a pick (Knowledge Demon's curse):
            # no one ends a turn with a pick open, so the log's `end_turn` is the
            # one the play caused.
            self.ci = self.next_choice + 1
            self.scan_choice()
        while (
            self.next_choice is not None
            and self.rows[self.next_choice]["t"] == "pick"
            and out["decision"] == "combat_play"
        ):
            # A pick with no real alternative: the host resolved it inside the action,
            # so what the log records after the pick has already happened there.
            row = self.rows[self.next_choice]
            self.note(
                f"auto_pick f{row.get('floor')} {[c.get('id') for c in row.get('cards', [])]}"
            )
            self.ci = self.next_choice + 1
            self.scan_choice()
        if t == "end_turn" and out["decision"] == "combat_play" and not self.hand_noted:
            # The new turn's hand: every card the log drew, plus at most the cards it
            # generated (the log does not say into which pile).
            drawn = Counter(x[1] for x in self.track.hand if x[2])
            made = Counter(x[1] for x in self.track.hand if not x[2])
            host = Counter(_norm(c["id"]) for c in out.get("hand", []))
            if drawn - host or (host - drawn) - made:
                self.hand_noted = True
                self.note(
                    f"hand_turn f{self.rows[self.combat_end - 1].get('floor')} round {out.get('round')}: "
                    f"host-only {sorted((host - drawn - made).elements())} "
                    f"log-only {sorted((drawn - host).elements())}"
                )
        host_in = out["decision"] == "combat_play" or (
            out["decision"] == "card_select" and bool(out.get("combat"))
        )
        if self.next_choice is None:
            if self.combat_end >= len(self.rows):
                raise LogEnd
            if host_in:
                return out  # the next step ends the turn (see the top of this method)
            return self.finish_combat(out)
        else:
            if not host_in:
                nxt = self.rows[self.next_choice]
                raise Desync(
                    "ended_early",
                    f"host left the fight, log goes on with {nxt['t']} {nxt.get('id', '')}",
                )
            # End-of-turn effects are logged before the player's `end_turn`.
            if self.rows[self.next_choice]["t"] != "end_turn" and out["decision"] == "combat_play":
                self.check_combat()
        return out

    def finish_combat(self, out: dict[str, Any]) -> dict[str, Any]:
        end = self.rows[self.combat_end]
        self.in_combat = False
        self.used.add(self.combat_end)
        self.combats += 1
        lost = out["decision"] == "game_over" and not out.get("victory")
        if (end.get("result") == "loss") != lost:
            raise Desync("combat_result", f"log {end.get('result')}, host {out['decision']}")
        if self.hits_lost and not lost:
            # Unknown from the log; the next logged hp reading checks the host's value.
            assert self.g is not None
            self.track.hp = self.g.get_state()["player"]["hp"]
            self.track.hp_unknown = False
            self.hits_lost = False
        return out

    # ---- outside combat ----------------------------------------------

    def find_decision(self, pred: Any) -> dict[str, Any] | None:
        for i in self.seg():
            r = self.rows[i]
            if i not in self.used and r["t"] == "decision" and pred(r):
                return r
        return None

    def use(self, dec: dict[str, Any]) -> None:
        self.used.add(self.dec_rows[dec["decision_id"]])

    def loose_pick(self) -> int | None:
        return next(
            (i for i in self.pending() if self.rows[i]["t"] == "pick"),
            None,
        )

    def on_event(self, d: dict[str, Any]) -> dict[str, Any]:
        if any(
            self.rows[i]["t"] == "room" and self.rows[i].get("id") == "CRYSTAL_SPHERE"
            for i in self.seg()
        ):
            raise Desync("unsupported", "the log does not hold the Crystal Sphere moves")
        opts = d.get("options", [])
        keys = [o.get("text_key") for o in opts]
        dec = self.find_decision(lambda r: r.get("decision_type") == "event")
        if dec is not None:
            ids = [o.get("option_id") for o in dec.get("options", [])]
            if {_loose(i) for i in ids} == {_loose(k) for k in keys}:
                self.use(dec)
                res = self.resolution(dec)
                # The same page can be logged again (the player looked at the map and
                # came back): the copies are one choice, answered once.
                for i in self.seg():
                    r = self.rows[i]
                    if (
                        i not in self.used
                        and r["t"] == "decision"
                        and r.get("event_id") == dec.get("event_id")
                        and r.get("page_index") == dec.get("page_index")
                        and [o.get("option_id") for o in r.get("options", [])] == ids
                    ):
                        self.used.add(i)
                        res = res or self.resolution(r)
                if res is None:
                    raise LogEnd
                want = _loose(res.get("option_id"))
                # Two options can share an id (one per potion slot): go by position first.
                oi = res.get("option_index")
                at = next((o for o in opts if o.get("index") == oi), None)
                same_order = [_loose(i) for i in ids] == [_loose(k) for k in keys]
                pick = (
                    at
                    if at is not None and same_order and _loose(at.get("text_key")) == want
                    else next((o for o in opts if _loose(o.get("text_key")) == want), None)
                )
                if pick is None:
                    raise Desync(
                        "event_options", f"chosen {res.get('option_id')} not in host {keys}"
                    )
                return self.act("choose_option", option_index=pick["index"])
            if len(opts) != 1:
                raise Desync("event_options", f"host {keys} log {ids}")
        open_opts = [o for o in opts if not o.get("is_locked")]
        if len(open_opts) == 1:
            return self.act("choose_option", option_index=open_opts[0]["index"])  # a proceed page
        if not self.pending() and self.k + 1 < len(self.segs):
            self.leave_room()
            return self.on_event(d)
        raise Desync("event_extra", f"host offers {keys}, log has no event decision here")

    def on_card_reward(self, d: dict[str, Any]) -> dict[str, Any]:
        cards = d.get("cards", [])
        host = [_card(c) for c in cards]
        dec = self.find_decision(
            lambda r: (
                r.get("decision_type") == "card_reward"
                and [_card(o) for o in r.get("options", [])] == host
            )
        )
        if dec is None:
            dec = self.find_decision(
                lambda r: (
                    r.get("decision_type") == "card_reward"
                    and sorted(_card(o)[0] for o in r.get("options", []))
                    == sorted(c[0] for c in host)
                )
            )
            if dec is not None:
                log_offer = [_card(o) for o in dec.get("options", [])]
                # An egg relic upgrades the cards of an open offer; the log has that as
                # `upgrade` rows after the offer, not in the offer itself.
                at = self.dec_rows[dec["decision_id"]]
                res0 = self.resolution(dec)
                stop = self.rows.index(res0) if res0 is not None else self.segs[self.k][1]
                for r in self.rows[at + 1 : stop]:
                    if r["t"] == "upgrade" and (_norm(r["id"]), False) in log_offer:
                        log_offer[log_offer.index((_norm(r["id"]), False))] = (_norm(r["id"]), True)
                if sorted(log_offer) != sorted(host):
                    raise Desync(
                        "card_reward_upgrade", f"host {sorted(host)} log {sorted(log_offer)}"
                    )
                self.note(f"card_reward order differs f{dec.get('floor')}: host {host}")
        alternatives = {a.get("id"): a["index"] for a in d.get("alternatives") or []}
        if dec is not None:
            did = dec["decision_id"]
            if did not in self.rerolled and any(
                r.get("decision_id") == did and r.get("outcome") == "reroll"
                for r in self.rows[self.segs[self.k][0] : self.segs[self.k][1] + 40]
                if r["t"] == "outcome"
            ):
                if "REROLL" not in alternatives:
                    raise Desync(
                        "card_reward_reroll",
                        "log rerolls the card reward; host offers no reroll",
                    )
                self.rerolled.add(did)
                # The first offer's row is answered by the reroll; the second offer
                # is matched when the host shows it.
                self.used.add(next(i for i in self.seg() if self.rows[i] is dec))
                return self.act("card_reward_alternative", alternative_index=alternatives["REROLL"])
            self.use(dec)
            res = self.resolution(dec)
            if res is not None and res["t"] == "acquire":
                # The host hands out a screen's card rewards in a fixed order; the
                # human may take them in another, which changes the deck's order.
                at = self.rows.index(res)
                for other in self.pending():
                    o = self.rows[other]
                    if o.get("decision_type") != "card_reward":
                        continue
                    ores = self.resolution(o)
                    if ores is not None and ores["t"] == "acquire" and self.rows.index(ores) < at:
                        raise Desync(
                            "reward_order",
                            f"log takes {ores.get('id')} before {res.get('id')}; host offers the latter first",
                        )
                # A card a thief took comes back as its own reward; the host takes it
                # before asking about the card reward, the human here took it after.
                back = next(
                    (
                        r
                        for r in self.rows[at + 1 : self.segs[self.k][1]]
                        if r["t"] == "acquire"
                        and (
                            r.get("source") == "granted"
                            or (r.get("decision_id") == dec["decision_id"] and r is not res)
                        )
                    ),
                    None,
                )
                if back is not None and not INTERACTIVE_REWARDS:
                    raise Desync(
                        "reward_order",
                        f"log takes {res.get('id')} before the returned {back.get('id')}; "
                        "host takes the returned card first",
                    )
            if res is not None and res["t"] == "acquire":
                want = _norm(res["id"])
                pick = next((c for c in cards if _norm(c["id"]) == want), None)
                if pick is None:
                    raise Desync("card_reward_offer", f"picked {want} not in host {host}")
                return self.act("select_card_reward", card_index=pick["index"])
            at = self.dec_rows[dec["decision_id"]]
            if any(
                r["t"] == "relic_counter" and r.get("id") == "PAELS_WING"
                for r in self.rows[at : self.segs[self.k][1]]
            ):
                if "SACRIFICE" not in alternatives:
                    raise Desync(
                        "card_reward_alternative",
                        "log sacrifices the reward to Pael's Wing; host offers no sacrifice",
                    )
                return self.act(
                    "card_reward_alternative", alternative_index=alternatives["SACRIFICE"]
                )
            return self.act("skip_card_reward")
        j = self.loose_pick()
        if j is not None:
            row = self.rows[j]
            wanted = [_norm(c["id"]) for c in row.get("cards", [])]
            if all(w in [h[0] for h in host] for w in wanted):
                self.used.add(j)
                if not wanted:
                    return self.act("skip_card_reward")
                pick = next(c for c in cards if _norm(c["id"]) == wanted[0])
                return self.act("select_card_reward", card_index=pick["index"])
        # Some logs hold the cards taken but not the offers (Kaleidoscope at the start).
        loose = [
            i
            for i in self.seg()
            if i not in self.used
            and self.rows[i]["t"] == "acquire"
            and self.rows[i].get("source") == "event"
            and not any(
                self.rows[i].get("c") == o.get("instance_id")
                for k in self.seg()
                if self.rows[k]["t"] == "decision"
                for o in self.rows[k].get("options", [])
            )
        ]
        hit = next((i for i in loose if _norm(self.rows[i]["id"]) in [h[0] for h in host]), None)
        if hit is not None:
            if hit != loose[0]:
                raise Desync(
                    "reward_order",
                    f"log takes {self.rows[loose[0]].get('id')} before {self.rows[hit].get('id')}; "
                    "host offers the latter first",
                )
            self.used.add(hit)
            pick = next(c for c in cards if _norm(c["id"]) == _norm(self.rows[hit]["id"]))
            return self.act("select_card_reward", card_index=pick["index"])
        logged = [
            [_card(o) for o in self.rows[i].get("options", [])]
            for i in self.seg()
            if self.rows[i]["t"] == "decision"
            and self.rows[i].get("decision_type") == "card_reward"
        ]
        raise Desync("card_reward_offer", f"host {host} log {logged}")

    def on_card_select(self, d: dict[str, Any]) -> dict[str, Any]:
        cards = d.get("cards", [])
        host = Counter(_card(c) for c in cards)

        def indices(wanted: list[tuple[str, bool]], ranks: list[int] | None = None) -> str:
            sel: list[int] = []
            for n, w in enumerate(wanted):
                same = [c for c in cards if _card(c) == w]
                free = [c for c in same if c["index"] not in sel]
                if not free:
                    raise Desync(
                        "card_select", f"{w} not among host options {sorted(host.elements())}"
                    )
                # Which of several identical copies: the same position as in the log,
                # since an enchanted or upgraded copy keeps its place in the deck.
                at = same[min(ranks[n], len(same) - 1)] if ranks else free[0]
                sel.append(at["index"] if at["index"] not in sel else free[0]["index"])
            return ",".join(map(str, sel))

        def rank_of(opts: list[dict[str, Any]], oi: Any) -> int:
            o = next((x for x in opts if x.get("option_index") == oi), None)
            if o is None:
                return 0
            return sum(
                1 for x in opts if _card(x) == _card(o) and x["option_index"] < o["option_index"]
            )

        dec = self.find_decision(lambda r: str(r.get("decision_type", "")).startswith(_DECK_SELECT))
        if (
            dec is not None
            and d.get("max_select") is not None
            and dec.get("max_select") not in (None, d["max_select"])
        ):
            other = self.find_decision(
                lambda r: (
                    str(r.get("decision_type", "")).startswith(_DECK_SELECT)
                    and r.get("max_select") == d["max_select"]
                )
            )
            if other is not None:
                # Two relics taken from one reward screen each ask for a pick; the
                # host takes the relics in its own order.
                raise Desync(
                    "reward_order",
                    f"host asks {other.get('decision_type')} before {dec.get('decision_type')}",
                )
        if dec is not None:
            opts = dec.get("options", [])
            if (
                Counter(_card(o) for o in opts if o.get("selectable", True)) != host
                and Counter(_card(o) for o in opts) != host
            ):
                later = self.find_decision(
                    lambda r: (
                        str(r.get("decision_type", "")).startswith(_DECK_SELECT)
                        and r is not dec
                        and len(r.get("options", [])) - len(cards) in (0, 1)
                        and r.get("max_select") == d.get("max_select", r.get("max_select"))
                    )
                )
                if later is not None:
                    # Two relics taken from one reward screen each ask for a pick; the
                    # host takes the relics in its own order.
                    raise Desync(
                        "reward_order",
                        f"host asks {later.get('decision_type')} before {dec.get('decision_type')}",
                    )
                raise Desync(
                    "card_select_options",
                    f"{dec.get('decision_type')}: host {sorted(host.elements())} "
                    f"log {sorted(_card(o) for o in opts)}",
                )
            self.use(dec)
            res = self.resolution(dec)
            if res is None:
                raise LogEnd
            if res.get("outcome") == "decline" or (res["t"] == "pick" and not res.get("cards")):
                return self.act("skip_select")
            if res["t"] == "pick":
                chosen = [_card(c) for c in res["cards"]]
                ois = [c.get("option_index") for c in res["cards"]]
            else:
                ois = list(res.get("selected_option_indices", []))
                chosen = [_card(opts[i]) for i in ois]
            ranks = [rank_of(opts, oi) for oi in ois]
            return self.act("select_cards", indices=indices(chosen, ranks))
        j = self.loose_pick()
        if j is not None:
            self.used.add(j)
            chosen = [_card(c) for c in self.rows[j].get("cards", [])]
            return (
                self.act("select_cards", indices=indices(chosen))
                if chosen
                else self.act("skip_select")
            )
        raise Desync(
            "card_select_extra",
            f"host asks to pick from {sorted(host.elements())}, log has no pick here",
        )

    def on_rest(self, d: dict[str, Any]) -> dict[str, Any]:
        dec = self.find_decision(lambda r: r.get("decision_type") == "rest")
        opts = d.get("options", [])
        if (
            dec is None
            and self.last_rest is not None
            and self.find_decision(
                lambda r: str(r.get("decision_type", "")).startswith(_DECK_SELECT)
            )
        ):
            # The human backed out of the card pick and chose the option again.
            again = next(
                (o for o in opts if str(o.get("option_id", "")).upper() == self.last_rest), None
            )
            if again is not None:
                return self.act("choose_option", option_index=again["index"])
        if dec is None:
            if d.get("can_leave"):
                return self.act("leave_room")
            raise Desync(
                "rest_extra",
                f"host offers {[o.get('option_id') for o in opts]}, log has no rest here",
            )
        self.use(dec)
        res = self.resolution(dec)
        if res is None:
            raise LogEnd
        want = str(res.get("option", "")).upper()
        self.last_rest = want
        pick = next((o for o in opts if str(o.get("option_id", "")).upper() == want), None)
        if pick is None:
            raise Desync(
                "rest_options", f"log chose {want}, host has {[o.get('option_id') for o in opts]}"
            )
        return self.act("choose_option", option_index=pick["index"])

    def on_shop(self, d: dict[str, Any]) -> dict[str, Any]:
        seen = next(
            (i for i in self.seg() if i not in self.used and self.rows[i]["t"] == "shop"), None
        )
        if seen is not None:
            row = self.rows[seen]
            for i in self.seg():
                if self.rows[i]["t"] == "shop":
                    self.used.add(i)
            for group in ("cards", "relics", "potions"):
                log_items = sorted(
                    (_norm(x.get("id", "?")), x.get("cost"))
                    for x in row.get(group, [])
                    if x.get("stocked")
                )
                host_items = sorted(
                    (
                        _norm(str(x.get("id") or x.get("entry") or "?"))
                        .removeprefix("RELIC.")
                        .removeprefix("POTION."),
                        x.get("cost"),
                    )
                    for x in d.get(group) or []
                    if x.get("is_stocked", True)
                )
                if log_items != host_items:
                    raise Desync("shop_stock", f"{group}: host {host_items} log {log_items}")
            if row.get("removal_stocked") and row.get("removal_cost") != d.get("card_removal_cost"):
                raise Desync(
                    "shop_stock",
                    f"removal: host {d.get('card_removal_cost')} log {row.get('removal_cost')}",
                )
        j = next((i for i in self.pending() if self.rows[i]["t"] == "buy"), None)
        if j is None:
            return self.act("leave_room")
        row = self.rows[j]
        self.used.add(j)
        kind = row.get("kind")
        if kind in ("card", "relic", "potion"):
            items = d.get(kind + "s") or []
            want = _norm(row.get("id", ""))
            hit = next(
                (
                    x
                    for x in items
                    if _norm(str(x.get("id") or x.get("entry") or ""))
                    .removeprefix("RELIC.")
                    .removeprefix("POTION.")
                    == want
                    and x.get("is_stocked", True)
                ),
                None,
            )
            if hit is None:
                raise Desync("shop_buy", f"{kind} {want} not stocked in host")
            return self.act(f"buy_{kind}", **{f"{kind}_index": hit["index"]})
        if "remov" in str(kind):
            return self.act("remove_card")
        raise Desync("unsupported", f"shop buy kind {kind}")

    def on_bundle(self, d: dict[str, Any]) -> dict[str, Any]:
        j = self.loose_pick()
        bundles = d.get("bundles", [])
        if j is None:
            raise Desync("bundle_extra", "host offers bundles, log has no pick here")
        want = Counter(_card(c)[0] for c in self.rows[j].get("cards", []))
        hit = next(
            (b for b in bundles if Counter(_card(c)[0] for c in b.get("cards", [])) == want), None
        )
        if hit is None:
            shown = [sorted(_card(c)[0] for c in b.get("cards", [])) for b in bundles]
            raise Desync(
                "bundle_options", f"log took {sorted(want.elements())}, host offers {shown}"
            )
        self.used.add(j)
        return self.act("select_bundle", bundle_index=hit["index"])

    def on_potion_swap(self, d: dict[str, Any]) -> dict[str, Any]:
        j = next((i for i in self.pending() if self.rows[i]["t"] == "potion_dropped"), None)
        drunk = next(
            (
                self.rows[i]
                for i in self.pending()
                if self.rows[i]["t"] == "potion_used" and self.rows[i - 1]["t"] != "outcome"
            ),
            None,
        )
        if drunk is not None:
            raise Desync(
                "potion_on_reward_screen",
                f"log drinks {drunk['id']} to make room for {d.get('potion')}; "
                "the host only offers to discard",
            )
        if j is None:
            # The host has already taken the screen's earlier potions by itself. When
            # the human left one of those and took this one, swap it in for that one.
            goal = Counter(self.track.belt)
            for r in self.rows[self.fed : self.segs[self.k][1]]:
                if r["t"] == "potion_got":
                    goal[r["id"]] += 1
                elif r["t"] in ("potion_used", "potion_dropped") and goal[r["id"]] > 0:
                    goal[r["id"]] -= 1
            held = [
                (o["discard_slot"], str(o.get("label", "")).removeprefix("discard "))
                for o in d.get("options", [])
                if o.get("discard_slot") is not None
            ]
            have = Counter(p for _, p in held)
            if goal[str(d.get("potion"))] > have[str(d.get("potion"))]:
                extra = next((slot for slot, p in held if have[p] > goal[p]), None)
                if extra is not None:
                    self.note(
                        f"potion_choice f{d.get('context', {}).get('total_floor')}: host took a potion the log left"
                    )
                    return self.act("potion_swap", discard_slot=extra)
            return self.act("potion_swap")
        want = self.rows[j]["id"]
        opt = next(
            (
                o
                for o in d.get("options", [])
                if o.get("discard_slot") is not None and want in json.dumps(o)
            ),
            None,
        )
        if opt is None:
            return self.act("potion_swap")
        self.used.add(j)
        return self.act("potion_swap", discard_slot=opt["discard_slot"])

    def screen_rows(self) -> range:
        """The log rows of the reward screen the host has open.

        After a fight they run from its `combat_end` to the next fight of the same
        room (act 2's boss floor has several), else to the room's end.
        """
        a, b = self.segs[self.k]
        start = self.combat_end + 1 if a <= self.combat_end < b else a
        end = next((i for i in range(start, b) if self.rows[i]["t"] == "combat_start"), b)
        return range(start, end)

    def reward_takes(self) -> list[int]:
        """This room's reward-screen rows not yet replayed, in the order the human acted.

        A taken reward is a `resolve` row (a card reward's follows its `acquire`),
        a chest relic an `outcome` row of the treasure decision; a potion drunk or
        dropped on the screen sits between them.
        """
        rows = self.rows
        screen = self.screen_rows()
        out = []
        for i in screen:
            r = rows[i]
            if i in self.used:
                continue
            taken = r["t"] == "resolve" and r.get("reward_kind") in _REWARD_TYPE
            chest = (
                r["t"] == "outcome"
                and r.get("decision_type") == "treasure"
                and r.get("outcome") == "select"
            )
            potion = r["t"] in ("potion_used", "potion_dropped") and rows[i - 1]["t"] != "outcome"
            if taken or chest or potion:
                out.append(i)

        def when(i: int) -> int:
            # A relic counts from its `relic` row: its `resolve` can follow a screen
            # the relic opened itself (Neow's Bones → Kaleidoscope's card rewards).
            r = rows[i]
            if r["t"] == "resolve" and r.get("reward_kind") == "RelicReward":
                for k in range(i - 1, screen.start - 1, -1):
                    if rows[k]["t"] == "relic" and rows[k].get("id") == r.get("id"):
                        return k
            return i

        return sorted(out, key=when)

    def on_rewards(self, d: dict[str, Any]) -> dict[str, Any]:
        """Take the rewards in the order the human took them, then leave the rest."""
        entries = d.get("rewards") or []
        screen = self.screen_rows()
        # The host lists a screen's card rewards in the game's order; the log has one
        # card_reward decision per card reward, in the same order (a reroll logs the
        # same decision again). Taken ones are used, so the rest line up.
        seen: set[int] = set()
        self.screen_cards = []
        for i in screen:
            r = self.rows[i]
            if (
                i not in self.used
                and r["t"] == "decision"
                and r.get("decision_type") == "card_reward"
                and r.get("source") == "reward"
                and r["decision_id"] not in seen
            ):
                seen.add(r["decision_id"])
                self.screen_cards.append(i)
        if self.screen_k != screen.start and not d.get("from_event"):
            # An event's screen can open another inside it (Neow's Bones →
            # Kaleidoscope), and some logs hold an event's cards but not its offers:
            # the count is checked on combat screens only.
            self.screen_k = screen.start
            n_host = sum(1 for e in entries if e.get("type") == "card")
            if n_host != len(self.screen_cards):
                raise Desync(
                    "rewards_offer",
                    f"host lists {n_host} card rewards, log {len(self.screen_cards)}: host "
                    + json.dumps([(e.get("type"), e.get("id")) for e in entries])
                    + " log "
                    + json.dumps([self.rows[i].get("decision_id") for i in self.screen_cards]),
                )
        takes = self.reward_takes()
        if not takes:
            # Everything left on the screen the human left too.
            for i in screen:
                if self.rows[i]["t"] == "decision" and self.rows[i].get("decision_type") in (
                    "relic_reward",
                    "treasure",
                ):
                    self.used.add(i)
            return self.act("proceed")
        j = takes[0]
        r = self.rows[j]
        if r["t"] in ("potion_used", "potion_dropped"):
            belt = d.get("player", {}).get("potions") or []
            slot = next((p for p in belt if p.get("entry") == r["id"]), None)
            if slot is None:
                raise Desync(
                    "potion", f"{r['id']} not on host belt {[p.get('entry') for p in belt]}"
                )
            self.used.add(j)
            action = "discard_potion" if r["t"] == "potion_dropped" else "use_potion"
            return self.act(action, potion_index=slot["index"])
        if r["t"] == "outcome":
            want_type, want_id = "relic", r.get("option_id")
        else:
            want_type, want_id = _REWARD_TYPE[r["reward_kind"]], r.get("id")
        cands = [e for e in entries if e.get("type") == want_type]
        if want_type in ("potion", "relic") and want_id:
            cands = [e for e in cands if e.get("id") == want_id]
        if want_type == "card":
            # Which card reward: the one offering the card instance the log's
            # `acquire` just before took (the log can file it under another
            # reward's decision), else the decision it names, else the card id.
            acq = next(
                (x for x in reversed(self.rows[screen.start : j]) if x["t"] == "acquire"),
                None,
            )
            row = None
            if acq is not None:
                for how in ("instance", "decision", "card"):
                    row = next(
                        (
                            i
                            for i in self.screen_cards
                            if (
                                how == "decision"
                                and self.rows[i]["decision_id"] == acq.get("decision_id")
                            )
                            or any(
                                (how == "instance" and o.get("instance_id") == acq.get("c"))
                                or (
                                    how == "card"
                                    and _norm(str(o.get("option_id"))) == _norm(acq["id"])
                                )
                                for o in self.rows[i].get("options", [])
                            )
                        ),
                        None,
                    )
                    if row is not None:
                        break
            if row in self.screen_cards:
                rank = self.screen_cards.index(row)
                cands = cands[rank : rank + 1]
            elif not self.screen_cards and len(cands) > 1:
                raise Desync(
                    "unsupported",
                    "the log holds the cards taken from this screen but not its offers; "
                    "which card reward came first is unknown",
                )
        if not cands:
            raise Desync(
                "rewards_offer",
                f"log takes {want_type} {want_id or ''}; host lists "
                + json.dumps([(e.get("type"), e.get("id")) for e in entries]),
            )
        self.used.add(j)
        if want_type == "relic":
            for i in screen:
                x = self.rows[i]
                if (
                    x["t"] == "decision"
                    and x.get("decision_type") in ("relic_reward", "treasure")
                    and any(o.get("option_id") == want_id for o in x.get("options", []))
                ):
                    self.used.add(i)
                    break
        return self.act("take_reward", reward_index=cands[0]["index"])

    def check_floor(self, d: dict[str, Any]) -> None:
        p = d.get("player") or {}
        t = self.track
        bad = []
        if not t.hp_seen and p.get("hp") is not None:
            t.hp, t.hp_seen = p["hp"], True  # the log's first hp reading is still to come
        if p.get("hp") != t.hp:
            bad.append(f"hp host {p.get('hp')} log {t.hp}")
        if p.get("max_hp") != t.max_hp:
            bad.append(f"max_hp host {p.get('max_hp')} log {t.max_hp}")
        if t.gold is not None and p.get("gold") != t.gold:
            bad.append(f"gold host {p.get('gold')} log {t.gold}")
        host_relics = Counter(r["id"] for r in p.get("relics") or [])
        wax = [r["id"] for r in p.get("relics") or [] if r.get("is_wax")]
        log_wax = [r for r in t.relics if r in wax]
        if not bad and host_relics == Counter(t.relics) and wax != log_wax:
            # Toy Box's wax relics melt in the order they were taken; the host takes
            # an event's reward screen in list order, the human chose another.
            raise Desync(
                "reward_order",
                f"wax relics taken in another order: host {wax} log {log_wax}",
            )
        if host_relics != Counter(t.relics):
            bad.append(
                f"relics host-only {sorted((host_relics - Counter(t.relics)).elements())} "
                f"log-only {sorted((Counter(t.relics) - host_relics).elements())}"
            )
        if "deck" in p:
            hd = Counter(_card(c) for c in p["deck"])
            ld = Counter(_card(c) for c in t.deck)
            a, b = self.segs[max(self.k - 1, 0)][0], self.segs[self.k][1]
            stolen = Counter(_norm(r["id"]) for r in self.rows[a:b] if r["t"] == "remove")
            back = Counter(
                _norm(r["id"])
                for r in self.rows[a:b]
                if r["t"] == "acquire" and r.get("source") == "granted"
            )
            extra = Counter(c[0] for c in (hd - ld).elements())
            if (
                not INTERACTIVE_REWARDS
                and not bad
                and extra
                and not ld - hd
                and not extra - (stolen - back)
            ):
                # A thief's card comes back as a reward the human may leave; the host
                # takes it without asking.
                raise Desync(
                    "host_took_declined_reward",
                    f"log leaves the stolen {sorted(extra.elements())} on the reward screen; host takes it",
                )
            if hd != ld:
                bad.append(
                    f"deck host-only {sorted((hd - ld).elements())} log-only {sorted((ld - hd).elements())}"
                )
        hb = Counter(x.get("entry") for x in p.get("potions") or [])
        if hb != Counter(t.belt):
            bad.append(f"potions host {sorted(hb.elements())} log {sorted(t.belt)}")
        if bad:
            raise Desync(
                "floor_state",
                f"after floor {self.rows[self.segs[self.k][1] - 1].get('floor')}: "
                + "; ".join(bad),
            )

    def leave_room(self) -> None:
        """Close the current room: nothing the human did may be left over."""
        for i in self.seg():
            r = self.rows[i]
            if i in self.used or r["t"] != "decision":
                continue
            if r.get("decision_type") in ("relic_reward", "treasure") and r.get("options"):
                res = self.resolution(r)
                if res is None or res.get("outcome") == "skip":
                    raise Desync(
                        "host_took_declined_reward",
                        f"{r['decision_type']} {[o.get('option_id') for o in r.get('options', [])]} left by the human",
                    )
        pend = self.pending()
        if pend:
            raise Desync("unconsumed", json.dumps(self.rows[pend[0]])[:300])
        self.feed_to(self.segs[self.k][1])
        if self.k + 1 >= len(self.segs):
            raise LogEnd
        self.k += 1

    def on_map(self, d: dict[str, Any]) -> dict[str, Any]:
        dec = next(
            (
                self.rows[i]
                for i in self.seg()
                if self.rows[i]["t"] == "decision"
                and self.rows[i].get("decision_type") == "map_node"
            ),
            None,
        )
        self.leave_room()
        self.check_floor(d)
        room = next(
            (
                self.rows[i]
                for i in self.seg()
                if self.rows[i]["t"] == "room" and self.rows[i].get("depth", 1) == 1
            ),
            None,
        )
        if room is None or not room.get("coord"):
            raise LogEnd
        col, row = (int(x) for x in room["coord"].split(","))
        choices = {(c["col"], c["row"]) for c in d.get("choices", [])}
        if dec is not None and not dec.get("free_travel"):
            logged = {
                tuple(int(x) for x in o["option_id"].split(",")) for o in dec.get("options", [])
            }
            if logged != choices:
                raise Desync("map_options", f"host {sorted(choices)} log {sorted(logged)}")
        if (col, row) not in choices:
            raise Desync("map_options", f"log went to {(col, row)}, host offers {sorted(choices)}")
        self.log(f"== floor {room.get('floor')} {room.get('room_type')} {room.get('id', '')}")
        return self.act("select_map_node", col=col, row=row)

    def loose_potion(self, d: dict[str, Any]) -> dict[str, Any] | None:
        """A potion drunk or dropped outside combat, if that is the log's next choice."""
        pend = self.pending()
        if not pend or self.rows[pend[0]]["t"] not in ("potion_used", "potion_dropped"):
            return None
        row = self.rows[pend[0]]
        if self.rows[pend[0] - 1]["t"] == "outcome":
            self.used.add(pend[0])  # the event took the potion, the player did not drop it
            return None
        belt = d.get("player", {}).get("potions") or []
        slot = next((p for p in belt if p.get("entry") == row["id"]), None)
        if slot is None:
            raise Desync("potion", f"{row['id']} not on host belt {[p.get('entry') for p in belt]}")
        self.used.add(pend[0])
        if row["t"] == "potion_dropped":
            return self.act("discard_potion", potion_index=slot["index"])
        return self.act("use_potion", potion_index=slot["index"])

    # ---- driver ------------------------------------------------------

    def step(self, d: dict[str, Any]) -> dict[str, Any]:
        kind = d["decision"]
        if kind == "combat_play" or (kind == "card_select" and d.get("combat")):
            if not self.in_combat:
                self.enter_combat(d)
            return self.combat_action(d)
        if self.in_combat:
            raise Desync("ended_early", f"host at {kind} while the log's fight goes on")
        if kind not in ("potion_swap", "rewards"):
            out = self.loose_potion(d)
            if out is not None:
                return out
        handler = {
            "map_select": self.on_map,
            "event_choice": self.on_event,
            "card_reward": self.on_card_reward,
            "card_select": self.on_card_select,
            "rest_site": self.on_rest,
            "shop": self.on_shop,
            "bundle_select": self.on_bundle,
            "potion_swap": self.on_potion_swap,
            "rewards": self.on_rewards,
        }.get(kind)
        if handler is None:
            raise Desync("unknown_host_decision", f"{kind}: {json.dumps(d)[:400]}")
        return handler(d)

    def run(self) -> dict[str, Any]:
        h = self.header
        out: dict[str, Any] = {
            "file": self.path,
            "seed": h.get("seed"),
            "character": h.get("character"),
            "ascension": h.get("ascension"),
            "log_floors": max((r.get("floor") or 0) for r in self.rows),
            "log_choices": sum(
                1 for r in self.rows if _is_combat_choice(r) or r["t"] in ("outcome", "buy", "rest")
            ),
        }
        neow = any(
            r["t"] == "decision" and r.get("event_id") == "NEOW"
            for r in self.rows[: self.segs[0][1]]
        )
        t0 = time.time()
        self.g = Game(forward_stderr=os.environ.get("HOST_REPLAY_STDERR") == "1")
        try:
            d = self.g.start(
                character=str(h["character"]).capitalize(),
                seed=h["seed"],
                ascension=h.get("ascension") or 0,
                act1="roll",
                played_save=not neow,
                interactive_potion_swap=True,
            )
            if d.get("type") != "decision":
                raise Desync("host_start", json.dumps(d)[:200])
            act = next((r.get("name") for r in self.rows[:12] if r["t"] == "act"), None)
            host_act = str(d.get("context", {}).get("act_name", "")).upper()
            if act and host_act != act:
                raise Desync("act", f"host {host_act} log {act}")
            st = self.g.get_state()
            hc = st.get("rng_counters") or {}
            rng = h.get("rng_state") or {}
            diff = {
                k: (hc[_STREAM.get(k, k)], v)
                for k, v in rng.items()
                if _STREAM.get(k, k) in hc and hc[_STREAM.get(k, k)] != v
            }
            if diff:
                self.note(f"rng@start host/log {diff}")
            out["rng_start_diff"] = {k: v[0] - v[1] for k, v in diff.items()}
            while True:
                self.log(f"host {d['decision']} (floor {d.get('context', {}).get('total_floor')})")
                if d["decision"] == "game_over":
                    end = next((r for r in self.rows if r["t"] == "end"), {})
                    left = sum(
                        1
                        for a, b in self.segs[self.k + 1 :]
                        for r in self.rows[a:b]
                        if r["t"] in ("combat_start", "buy", "rest", "pick")
                        or (r["t"] == "decision" and r.get("event_id") != "THE_ARCHITECT")
                    )
                    # The closing dialogue after the last boss has one option a page;
                    # the host walks it without the log's rows being matched.
                    pend = [
                        i for i in self.pending() if self.rows[i].get("event_id") != "THE_ARCHITECT"
                    ]
                    if pend or left:
                        raise Desync(
                            "host_game_over",
                            f"host ended (victory={d.get('victory')}), log goes on: "
                            f"{left} choices in later rooms, pending "
                            + json.dumps([self.rows[i] for i in pend[:2]])[:300],
                        )
                    out["end"] = (
                        f"game_over victory={d.get('victory')} log={end.get('terminal_reason')}"
                    )
                    break
                d = self.step(d)
        except LogEnd:
            out["end"] = "log_end"
        except Desync as e:
            out["desync"] = e.kind
            out["detail"] = e.detail[:600]
        except Exception as e:
            out["desync"] = "harness:" + type(e).__name__
            out["detail"] = str(e)[:400]
        finally:
            try:
                errs = self.g.world_errors(drain_timeout=0.2)
                out["world_errors"] = errs.get("count", 0) if isinstance(errs, dict) else 0
            except Exception:
                pass
            self.g.close()
        out["floor"] = self.rows[min(self.fed, len(self.rows)) - 1].get("floor")
        out["rooms"] = self.k
        out["actions"] = self.actions
        out["combats"] = self.combats
        out["notes"] = self.notes
        out["secs"] = round(time.time() - t0, 1)
        return out


def replay_file(path: str) -> dict[str, Any]:
    try:
        rows = load(path)
    except (OSError, EOFError) as e:
        return {"file": path, "skip": f"unreadable:{type(e).__name__}"}
    why = eligible(rows)
    if why:
        return {"file": path, "skip": why}
    return Replayer(path).run()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    ap.add_argument("--run", help="replay one file, verbosely")
    ap.add_argument("--census", type=int, default=0, help="number of eligible runs to replay")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--character", help="restrict to one character directory")
    ap.add_argument("--out", help="directory for results.jsonl and summary.json")
    args = ap.parse_args()
    set_game_version(GAME_VERSION)

    if args.run:
        why = eligible(load(args.run))
        if why:
            print(f"not eligible: {why}")
        print(json.dumps(Replayer(args.run, verbose=True).run(), indent=1))
        return

    pattern = f"{replays_root()}/{args.character or '*'}/*/*.replay.ndjson.gz"
    files = sorted(glob.glob(pattern))
    random.Random(args.seed).shuffle(files)
    out_dir = Path(args.out or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    skips: Counter[str] = Counter()
    with (
        Pool(args.workers) as pool,
        (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh,
    ):
        for res in pool.imap_unordered(replay_file, files):
            if "skip" in res:
                skips[res["skip"]] += 1
                continue
            results.append(res)
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(
                f"[{len(results)}] {res['character']} A{res['ascension']} rooms {res['rooms']} "
                f"actions {res['actions']} -> {res.get('desync') or res.get('end')}",
                flush=True,
            )
            if len(results) >= args.census:
                pool.terminate()
                break
    kinds = Counter(
        r.get("desync") or "in_sync:" + str(r.get("end", "")).split(" ")[0] for r in results
    )
    summary = {
        "runs": len(results),
        "skipped": dict(skips),
        "outcome": dict(kinds.most_common()),
        "rooms_median": sorted(r["rooms"] for r in results)[len(results) // 2] if results else 0,
        "actions_total": sum(r["actions"] for r in results),
        "combats_in_sync": sum(r["combats"] for r in results),
        "world_errors": sum(r.get("world_errors", 0) for r in results),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
