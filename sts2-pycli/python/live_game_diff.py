"""Live same-seed diff: the headless host and the running game in lockstep (GH #1287).

The real game (through the Spire Mentor mod's TCP bridge, :class:`sts2_realgame.
RealGame`) and the headless host (:class:`sts2_client.Game`) start the same seed,
character and ascension. At every decision the policy reads the HOST's decision
and picks one command; both worlds get that command, and the two decisions that
come back are compared before the next pick. The first decision on which they
disagree ends the run: from there on a command picked for the host may mean
something else in the game.

The policy is the live decision stack of :func:`headless_eval.
play_one_headless_session` (the PPO champion for run-level choices, the combat
MCTS for fights), so runs reach acts 2 and 3. It only ever sees the host, so a
choice cannot differ between the two worlds; only their responses can.

What counts as a disagreement:

- the decision kind differs (``combat_play`` on one side, ``map_select`` on the
  other), or one side errors;
- a field that both sides send has different values. Fields that only one side
  sends are wire shape (the bridge does not carry them) and are counted, not
  stopped on. :data:`IGNORE` lists the fields the two wires fill differently on
  purpose; each entry says why.

Usage (from ``sts2-pycli/``, game running with the mod's agent lane on)::

    python python/live_game_diff.py --characters Ironclad --ascensions 0 \\
        --seeds 1000 --runs 1 --out ../docs/experiments/<date>-live-game-diff

The game's mod must run with ``auto_start: true``, ``client_run_config: true``
and ``advisor_autoplay: false`` (``SpireMentor.conf``), so it waits for a
``start_run`` from this client. A run that diverges is left mid-way in the
game; :func:`restart_game` kills and relaunches it, and the mod's autopilot
abandons the saved run on boot.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import sts2_client
from realgame_parity import flatten_decision
from sts2_client import Game
from sts2_realgame import RealGame

logger = logging.getLogger(__name__)

#: Field paths (list indices collapsed to ``[]``) the two wires fill
#: differently by construction. Matched with ``fnmatch``.
IGNORE: tuple[str, ...] = (
    # RealGame cannot read these from the bridge (sts2_realgame docstring).
    "context.act_name",
    "context.boss",
    "context.boss.*",
    "act_name",
    "*.stats",
    "*.after_upgrade",
    "card_reward.gold_earned",
    "event_name",
    # Rule text: the host sends the localization template ("Deal
    # {Damage:diff()} damage."), the game the rendered BBCode ("Deal 6
    # damage."). The numbers behind it are compared through `stats`/`vars`
    # wherever both wires carry them.
    "*description",
    # The game numbers pile cards after the hand (5, 6, ...), the host from
    # 0. The pile's order is still compared, through the ids at each
    # position.
    "*_pile[].index",
    # Rest-site option label: the host sends the class name
    # ("HealRestSiteOption"), the game the button text ("Rest"). Both send
    # the same `option_id` ("HEAL"), which is compared.
    "options[].name",
)

#: Fields RealGame fills with ``None`` when the bridge message at hand does not
#: carry them. ``None`` there means "not sent", any other value is compared.
REAL_MAY_OMIT: frozenset[str] = frozenset(
    {
        # card_reward / reward messages carry no room (RealGame._context).
        "context.room_type",
    }
)


#: ``RunSimulator.CrystalSphereSweepCenters``: the cells the host digs with
#: the Big tool, in order (all six only on the 6-divination option).
HOST_CRYSTAL_SPHERE_CENTERS: tuple[tuple[int, int], ...] = (
    (1, 3),
    (4, 3),
    (7, 3),
    (1, 6),
    (4, 6),
    (7, 6),
)


def _is_crystal_sphere(decision: dict[str, Any]) -> bool:
    """The game's dig screen, as RealGame translates it."""
    return _kind(decision) == "event_choice" and any(
        str(o.get("title", "")).startswith("Cell (") for o in decision.get("options") or []
    )


class LockstepDivergence(Exception):
    """The host and the game answered the same command differently."""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__(f"{record['kind']} at step {record['step']}: {record.get('summary')}")
        self.record = record


def _kind(decision: dict[str, Any]) -> str:
    if decision.get("type") == "decision":
        return str(decision.get("decision"))
    return str(decision.get("type"))


def _ignored(norm: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(norm, p) for p in patterns)


_UPGRADE_SUFFIX = re.compile(r"\+\d*$")
_UNLOCALIZED = re.compile(r"^[A-Z0-9_]+\.(?:title|name)$")
_PLACEHOLDER = re.compile(r"\{[A-Za-z]\w*\}")


def _unformatted(host: Any, real: Any) -> bool:
    """The host sent the text template the game fills in ("Insert {Rarity} Potion")."""
    if not (isinstance(host, str) and isinstance(real, str) and _PLACEHOLDER.search(host)):
        return False
    parts = _PLACEHOLDER.split(host)
    return re.fullmatch(".+?".join(map(re.escape, parts)), real) is not None


_ID_PREFIX = re.compile(r"^(CARD|RELIC|POTION|POWER|MONSTER|ENCHANTMENT|ORB)\.")
_EMPTY = (None, "<empty-list>", "<empty-dict>")


def _same(norm: str, host: Any, real: Any) -> bool:
    """Equal, or equal once a wire's display convention is undone.

    - Card names: the game shows an upgraded card as "Strike+" (the host
      sends "Strike" with ``upgraded: true``, which is compared separately).
    - Ids: the host's combat wire prefixes the model kind ("CARD.BASH"), the
      bridge's does not ("BASH").
    - Empty: an empty pile is ``null`` on one wire and ``[]`` on the other.
    - Enchantments: the host sends the name ("Swift"), the bridge the id
      ("SWIFT").
    """
    if host == real:
        return True
    if host in _EMPTY and real in _EMPTY:
        return True
    if norm.endswith("id") and isinstance(host, str) and isinstance(real, str):
        return _ID_PREFIX.sub("", host) == _ID_PREFIX.sub("", real)
    if norm.endswith(".enchantment") and isinstance(host, str) and isinstance(real, str):
        # The host names it ("Perfect Fit"), the bridge sends its id ("PERFECT_FIT").
        return host.upper().replace(" ", "_") == real.upper().replace(" ", "_")
    if norm.endswith(".name") and isinstance(host, str) and isinstance(real, str):
        return _UPGRADE_SUFFIX.sub("", host) == _UPGRADE_SUFFIX.sub("", real)
    return False


_SHOP_LISTS = ("cards", "relics", "potions")


def _sold_out_blanked(host: dict[str, Any], real: dict[str, Any]) -> tuple[dict, dict]:
    """Reduce shop slots both sides call sold out to that one fact.

    The host keeps a bought slot with its card cleared (name ``'?.title'``,
    type ``'?'``); the game keeps the item. Neither can be bought again.
    """
    if _kind(host) != "shop" or _kind(real) != "shop":
        return host, real
    host, real = dict(host), dict(real)
    for key in _SHOP_LISTS:
        hs, rs = list(host.get(key) or []), list(real.get(key) or [])
        for i in range(min(len(hs), len(rs))):
            if hs[i].get("is_stocked") is False and rs[i].get("is_stocked") is False:
                hs[i] = rs[i] = {"is_stocked": False}
        host[key], real[key] = hs, rs
    return host, real


def _map_choices_sorted(decision: dict[str, Any]) -> dict[str, Any]:
    """Map choices in (row, col) order: the wires list them in different orders,
    and a move names its node by position, not by list index."""
    if _kind(decision) != "map_select" or not decision.get("choices"):
        return decision
    choices = sorted(decision["choices"], key=lambda c: (c.get("row", 0), c.get("col", 0)))
    return {**decision, "choices": choices}


#: The card keywords the game's side can express. In combat realgame_adapter
#: rebuilds a card's keywords from four bridge flags
#: (`_keywords_for_hand_card`); everywhere else RealGame does the same from
#: four others (`sts2_realgame._KEYWORD_FLAGS`). Eternal and the rest never
#: reach the game's side.
_COMBAT_KEYWORDS = frozenset({"retain", "ethereal", "innate", "unplayable"})
_CARD_KEYWORDS = frozenset({"exhaust", "innate", "retain", "ethereal"})
_PILES = frozenset({"hand", "draw_pile", "discard_pile", "exhaust_pile", "play_pile"})


def _bridge_keywords(node: Any, allowed: frozenset[str] = _CARD_KEYWORDS) -> Any:
    """Every card's keywords cut down to what the game's side can carry."""
    if isinstance(node, list):
        return [_bridge_keywords(x, allowed) for x in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key == "keywords" and (value is None or isinstance(value, list)):
            kws = {str(k).lower() for k in value or []} & allowed
            out[key] = sorted(kws) or None
        else:
            out[key] = _bridge_keywords(value, _COMBAT_KEYWORDS if key in _PILES else allowed)
    return out


def _run_floor(decision: dict[str, Any]) -> dict[str, Any]:
    """The host's floor counted over the whole run, as the game counts it.

    The host's ``floor`` (``RunState.ActFloor``, in ``context`` and at the top
    of a map choice) restarts every act; ``context.total_floor`` does not.
    """
    ctx = decision.get("context")
    if not isinstance(ctx, dict) or ctx.get("total_floor") is None:
        return decision
    total = ctx["total_floor"]
    out = {**decision, "context": {**ctx, "floor": total}}
    if "floor" in decision:
        out["floor"] = total
    return out


def _dead_split(decision: dict[str, Any]) -> dict[str, Any]:
    """The game's enemy list laid out the way the host sends it.

    The bridge keeps corpses inline in ``enemies`` (``is_dead``) and numbers
    every creature; the host lists the living in ``enemies``, numbered among
    themselves, and the corpses apart in ``dead_enemies`` without a number.
    """
    enemies = decision.get("enemies")
    if not isinstance(enemies, list) or not any(
        isinstance(e, dict) and e.get("is_dead") for e in enemies
    ):
        return decision
    alive = [e for e in enemies if not (isinstance(e, dict) and e.get("is_dead"))]
    dead = [e for e in enemies if isinstance(e, dict) and e.get("is_dead")]
    return {
        **decision,
        "enemies": [{**e, "index": i} for i, e in enumerate(alive)],
        "dead_enemies": [{k: v for k, v in e.items() if k != "index"} for e in dead],
    }


def _game_target(real: dict[str, Any], target: Any) -> Any:
    """The game's index of the host's ``target``-th living enemy."""
    enemies = real.get("enemies")
    if not isinstance(target, int) or not isinstance(enemies, list):
        return target
    alive = [
        e.get("index", i)
        for i, e in enumerate(enemies)
        if isinstance(e, dict) and not e.get("is_dead")
    ]
    return alive[target] if 0 <= target < len(alive) else target


def _bridge_blind_spots(host: dict[str, Any], real: dict[str, Any]) -> tuple[dict, dict]:
    """Drop what the bridge cannot report, so it is not read as a difference.

    * ``player.block`` outside ``combat_play``: RealGame builds that player
      from ``run_state``, which carries no block, and always says 0.
    * ``min_select`` of a choose-a-card screen: the game hands the mod's
      selector ``(cards, 0, 1)`` whatever ``canSkip`` says
      (``CardSelectCmd.FromChooseACardScreen``), so the mod reports 0 where
      the real screen offers no skip. The host's number follows ``canSkip``.
    """
    if _kind(host) != "combat_play":
        host, real = _without_block(host), _without_block(real)
    if host.get("source") == "FromChooseACardScreen" and real.get("min_select") == 0:
        real = {**real, "min_select": host.get("min_select")}
    return host, real


def _without_block(decision: dict[str, Any]) -> dict[str, Any]:
    player = decision.get("player")
    if not isinstance(player, dict) or "block" not in player:
        return decision
    return {**decision, "player": {k: v for k, v in player.items() if k != "block"}}


def _x_cost_blind(decision: dict[str, Any]) -> dict[str, Any]:
    """X-cost cards in combat piles, with ``cost`` left out of the comparison.

    Neither side's number is the card's cost. ``realgame_adapter`` sends an X
    card's ``effective_cost`` (the player's energy) because the engine plays X
    off it; the host says 0 in hand and the X last paid once played (4 in the
    discard pile after Heavenly Drill took 4 energy, where the game says 0).
    ``base_cost`` -1 marks an X card on both sides.
    """
    out = dict(decision)
    for key in _PILES:
        pile = decision.get(key)
        if isinstance(pile, list):
            out[key] = [
                {k: v for k, v in c.items() if k != "cost"}
                if isinstance(c, dict) and c.get("base_cost") == -1
                else c
                for c in pile
            ]
    return out


def _buyable(shop: dict[str, Any]) -> bool:
    """Whether the host's shop has anything the player can pay for now."""
    gold = (shop.get("player") or {}).get("gold") or 0
    for key in _SHOP_LISTS:
        for item in shop.get(key) or []:
            if item.get("is_stocked") and (item.get("cost") or 0) <= gold:
                return True
    cost = shop.get("card_removal_cost")
    return cost is not None and cost <= gold


def compare(
    host: dict[str, Any], real: dict[str, Any], *, ignore: tuple[str, ...] = IGNORE
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split the differences between two decisions into content and wire shape.

    Returns:
        ``(content, shape)``. ``content`` holds one entry per field both sides
        send with different values (``path``, ``host``, ``real``), plus a
        ``kind`` entry when the decision kinds differ. ``shape`` holds the
        normalized paths that only one side sends.
    """
    content: list[dict[str, Any]] = []
    kh, kr = _kind(host), _kind(real)
    if kh != kr:
        content.append({"path": "<kind>", "host": kh, "real": kr})
        return content, []
    host, real = _sold_out_blanked(host, real)
    host, real = _map_choices_sorted(host), _map_choices_sorted(real)
    host, real = _bridge_keywords(host), _bridge_keywords(real)
    host = _run_floor(host)
    real = _x_cost_blind(_dead_split(real))
    host = _x_cost_blind(host)
    host, real = _bridge_blind_spots(host, real)
    h = {p: (n, v) for p, n, v in flatten_decision(host)}
    r = {p: (n, v) for p, n, v in flatten_decision(real)}
    shape: set[str] = set()
    for path in sorted(set(h) | set(r)):
        if path in h and path in r:
            norm, hv = h[path]
            rv = r[path][1]
            if _same(norm, hv, rv) or _ignored(norm, ignore):
                continue
            if rv is None and norm in REAL_MAY_OMIT:
                shape.add(f"host-only:{norm}")
                continue
            if isinstance(hv, str) and _UNLOCALIZED.match(hv):
                # The host could not resolve a display name and sent the key
                # ("SPEED_POTION_POWER.title"). Display text only: noted.
                shape.add(f"host-unlocalized:{norm}={hv}")
                continue
            if _unformatted(hv, rv):
                shape.add(f"host-unformatted:{norm}={hv}")
                continue
            content.append({"path": path, "host": hv, "real": rv})
        else:
            norm = (h.get(path) or r.get(path))[0]  # type: ignore[index]
            if not _ignored(norm, ignore):
                shape.add(f"{'host' if path in h else 'real'}-only:{norm}")
    return content, sorted(shape)


class LockstepGame:
    """A ``sts2_client.Game`` stand-in that plays the host and the game together.

    Every command goes to the host first. A command the host rejects is not
    sent to the game (the game never saw the bad pick), and the ``proceed``
    the policy sends after a rejection is answered by the host alone. Every
    other reply pair is compared; a content difference raises
    :class:`LockstepDivergence` after the step is written to the trail.
    """

    def __init__(
        self,
        host: Game,
        real: RealGame,
        trail: IO[str] | None = None,
        *,
        ignore: tuple[str, ...] = IGNORE,
    ) -> None:
        self.host = host
        self.real = real
        self.trail = trail
        self.ignore = ignore
        self.step = 0
        self.shape_counts: Counter[str] = Counter()
        self.divergence: dict[str, Any] | None = None
        self.last_floor: Any = None
        self.last_act: Any = None
        self._host_rejected = False
        self._last_real: dict[str, Any] = {}
        # The game's reply already received for a shop the mod skipped
        # because nothing in it could be bought; it answers the host's
        # `leave_room` (see `_check`).
        self._skipped_shop_reply: dict[str, Any] | None = None

    # sts2_client.Game surface used by headless_eval ------------------------

    @property
    def proc(self) -> Any:
        return self.host.proc

    def world_errors(self) -> dict[str, Any]:
        return self.host.world_errors()

    def get_map(self) -> dict[str, Any]:
        return self.host.get_map()

    def start(
        self,
        character: str = "Ironclad",
        seed: str = "test",
        ascension: int = 0,
        played_save: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        h = self.host.start(character, seed=seed, ascension=ascension, played_save=played_save)
        r = self.real.start(
            character=character, seed=int(seed), ascension=ascension, require_new_run=True
        )
        self._check({"cmd": "start_run", "seed": seed}, h, r)
        return h

    def act(self, action: str, **args: Any) -> dict[str, Any]:
        cmd: dict[str, Any] = {"cmd": "action", "action": action}
        non_null = {k: v for k, v in args.items() if v is not None}
        if non_null:
            cmd["args"] = non_null
        return self.send(cmd)

    def send(self, cmd: dict[str, Any]) -> dict[str, Any]:
        if cmd.get("cmd") != "action":
            return self.host.send(cmd)
        if self._skipped_shop_reply is not None and cmd.get("action") != "leave_room":
            r = self._skipped_shop_reply
            self._skipped_shop_reply = None
            h = self.host.send(cmd)
            self._diverge(
                cmd,
                h,
                r,
                [{"path": "<shop>", "host": cmd.get("action"), "real": "left: nothing to buy"}],
            )
        h = self.host.send(cmd)
        if h.get("type") == "error":
            self._host_rejected = True
            self._write(cmd, h, None, [], [], note="host rejected; not sent to the game")
            return h
        if self._host_rejected and cmd.get("action") == "proceed":
            self._host_rejected = False
            r = self._last_real
        elif self._skipped_shop_reply is not None:
            r = self._skipped_shop_reply
            self._skipped_shop_reply = None
        else:
            self._host_rejected = False
            r = self._play_crystal_sphere(h, self.real.send(self._for_game(cmd)))
        self._check(cmd, h, r)
        return h

    # internals --------------------------------------------------------------

    def _play_crystal_sphere(self, h: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
        """Dig the game's Crystal Sphere the way the host digs it, then leave.

        The host plays the minigame itself and goes on to its rewards
        (``RunSimulator.PlayCrystalSphereHeadless``: the Big tool on
        :data:`HOST_CRYSTAL_SPHERE_CENTERS`, in order); the game asks for each
        cell. Choosing a cell draws no RNG, so the same clicks must bring the
        same rewards, and the reply after ``proceed`` is compared with the
        host's as usual.
        """
        centers = list(HOST_CRYSTAL_SPHERE_CENTERS)
        while _is_crystal_sphere(r) and not _is_crystal_sphere(h):
            options = {o.get("title"): o.get("index") for o in r.get("options") or []}
            # The game offers proceed only once the divinations are spent.
            pick = options.get("Proceed")
            while pick is None and centers:
                x, y = centers.pop(0)
                pick = options.get(f"Cell ({x},{y})")
            if pick is None:
                return r  # nothing the host would click: let the comparison show it
            cmd = {"cmd": "action", "action": "choose_option", "args": {"option_index": pick}}
            self._write(cmd, h, r, [], [], note="game's Crystal Sphere dug the host's way")
            r = self.real.send(cmd)
        return r

    def _for_game(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """``cmd`` with a host target number turned into the game's.

        The host numbers living enemies only; the game numbers every creature
        in combat, corpses included, so after a death the same number names
        a different enemy.
        """
        args = cmd.get("args") or {}
        if "target_index" not in args:
            return cmd
        target = _game_target(self._last_real, args["target_index"])
        return {**cmd, "args": {**args, "target_index": target}}

    def _check(self, cmd: dict[str, Any], h: dict[str, Any], r: dict[str, Any]) -> None:
        self._last_real = r
        ctx = h.get("context") or {}
        self.last_floor = ctx.get("floor", self.last_floor)
        self.last_act = ctx.get("act", self.last_act)
        if _kind(h) == "shop" and _kind(r) != "shop" and not _buyable(h):
            # The mod leaves a shop with nothing to buy without asking
            # (SimShopRoomHandler.ChooseShopOption); the host still asks. Its
            # reply waits for the host's `leave_room`.
            self._skipped_shop_reply = r
            self._write(cmd, h, r, [], [], note="game skipped a shop with nothing to buy")
            return
        content, shape = compare(h, r, ignore=self.ignore)
        self.shape_counts.update(shape)
        self._write(cmd, h, r, content, shape)
        if content:
            self._diverge(cmd, h, r, content, written=True)

    def _diverge(
        self,
        cmd: dict[str, Any],
        h: dict[str, Any],
        r: dict[str, Any],
        content: list[dict[str, Any]],
        *,
        written: bool = False,
    ) -> None:
        if not written:
            self._write(cmd, h, r, content, [])
        raw = getattr(self.real, "last_raw", None)
        if self.trail is not None and raw is not None:
            # The bridge message behind the game's side, untranslated.
            row = {"step": self.step - 1, "real_raw": raw}
            self.trail.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            self.trail.flush()
        self.divergence = {
            "step": self.step - 1,
            "kind": "kind" if content[0]["path"] == "<kind>" else "content",
            "decision": _kind(h),
            "act": self.last_act,
            "floor": self.last_floor,
            "command": cmd,
            "summary": "; ".join(
                f"{d['path']}: host={d['host']!r} real={d['real']!r}" for d in content[:6]
            ),
            "diffs": content[:40],
        }
        raise LockstepDivergence(self.divergence)

    def _write(
        self,
        cmd: dict[str, Any],
        h: dict[str, Any],
        r: dict[str, Any] | None,
        content: list[dict[str, Any]],
        shape: list[str],
        note: str | None = None,
    ) -> None:
        row = {"step": self.step, "cmd": cmd, "host": h, "real": r, "content": content}
        if shape:
            row["shape"] = shape
        if note:
            row["note"] = note
        self.step += 1
        if self.trail is not None:
            self.trail.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            self.trail.flush()


# ── Game process control (Windows, Steam) ─────────────────────────────────

GODOT_LOG = Path(os.environ.get("APPDATA", "")) / "SlayTheSpire2" / "logs" / "godot.log"
STEAM_APPID = "2868840"


def _steam_exe() -> str:
    out = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-ItemProperty 'HKCU:\\Software\\Valve\\Steam').SteamExe",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return out.stdout.strip()


#: The mod logs this once it is waiting for our ``start_run``.
READY_MARKER = b'client_run_config: waiting for agent "start_run"'


def _log_has(marker: bytes) -> bool:
    try:
        return marker in GODOT_LOG.read_bytes()
    except OSError:
        return False


def restart_game(timeout: float = 240.0, game_dir: Path | None = None) -> None:
    """Kill the game, relaunch it, wait for the mod to ask for a run.

    Through Steam by default. With ``game_dir``, that copy's executable is
    started directly: a build fetched with ``download_depot`` that Steam
    would otherwise update on launch (Steam must still be running).
    """
    subprocess.run(
        ["taskkill", "/F", "/IM", "SlayTheSpire2.exe"],
        capture_output=True,
        check=False,
    )
    time.sleep(5)
    # Move the old log aside, so the ready marker of the previous launch
    # cannot satisfy the wait below.
    try:
        GODOT_LOG.replace(GODOT_LOG.with_name("godot.before-restart.log"))
    except OSError:
        pass
    if game_dir is not None:
        (game_dir / "steam_appid.txt").write_text(STEAM_APPID, encoding="utf-8")
        exe = game_dir / "SlayTheSpire2.exe"
        launch = f"Start-Process -FilePath '{exe}' -WorkingDirectory '{game_dir}'"
    else:
        launch = (
            f"Start-Process -FilePath '{_steam_exe()}' -ArgumentList '-applaunch','{STEAM_APPID}'"
        )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", launch],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(3)
        if _log_has(READY_MARKER):
            return
    raise RuntimeError(f"the game did not come back within {timeout:.0f}s")


# ── Sweep ──────────────────────────────────────────────────────────────────


@dataclass
class RunRow:
    character: str
    ascension: int
    seed: int
    result: str
    steps: int
    act: Any
    floor: Any
    duration_s: float
    divergence: dict[str, Any] | None = None
    shape_counts: dict[str, int] = field(default_factory=dict)
    world_errors: int | None = None
    #: The Rust combat mirror's own misses (it plays the host's fights for the
    #: search). Not a host-vs-game difference: recorded, never fatal here.
    mirror_stats: dict[str, Any] | None = None


def play_one(
    *,
    character: str,
    ascension: int,
    seed: int,
    out_dir: Path,
    host: str,
    port: int,
    max_steps: int,
    max_run_seconds: float,
    sims_regular: int,
    sims_boss: int,
    ppo_policies: Any,
) -> RunRow:
    """Play one seed in lockstep and write its trail. Returns the result row."""
    from headless_eval import play_one_headless_session
    from sts2_hint.recommender import RecommendConfig

    trail_path = out_dir / "trails" / f"{character.lower()}_a{ascension}_{seed}.jsonl"
    trail_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with trail_path.open("w", encoding="utf-8") as trail:
        real = RealGame(host=host, port=port, character=character, seed=seed, ascension=ascension)
        real.connect()
        lockstep = LockstepGame(Game(), real, trail)
        try:
            outcome = play_one_headless_session(
                random.Random(seed),
                run_index=0,
                character=character,
                seed=seed,
                mcts_cfg=RecommendConfig(
                    character=character,
                    hidden_info="beliefdraw",
                    c_puct=2.5,
                    draw_widening_c=2.0,
                ),
                ppo_policies=ppo_policies,
                sims_regular=sims_regular,
                sims_boss=sims_boss,
                max_steps=max_steps,
                max_run_seconds=max_run_seconds,
                ascension=ascension,
                game=lockstep,
            )
        finally:
            real.close()
    if lockstep.divergence is not None:
        result = "diverged"
    elif outcome.result in ("victory", "defeat"):
        result = outcome.result
    else:
        result = outcome.result
    return RunRow(
        character=character,
        ascension=ascension,
        seed=seed,
        result=result,
        steps=lockstep.step,
        act=lockstep.last_act,
        floor=lockstep.last_floor,
        duration_s=round(time.monotonic() - started, 1),
        divergence=lockstep.divergence,
        shape_counts=dict(lockstep.shape_counts),
        world_errors=(outcome.world_errors or {}).get("count"),
        mirror_stats=outcome.mirror_stats,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--characters", default="Ironclad,Silent,Defect,Regent,Necrobinder")
    parser.add_argument("--ascensions", default="0,10")
    parser.add_argument("--seeds", type=int, default=1000, help="first seed")
    parser.add_argument("--runs", type=int, default=10, help="seeds per character and ascension")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--max-run-seconds", type=float, default=5400.0)
    parser.add_argument("--sims-regular", type=int, default=1000)
    parser.add_argument("--sims-boss", type=int, default=4000)
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="do not relaunch the game after a run that ends mid-way",
    )
    parser.add_argument(
        "--game-dir",
        type=Path,
        help="launch this copy of the game directly instead of through Steam",
    )
    parser.add_argument(
        "--game-version",
        default="v0.111.0",
        help="the running game's build; the host must be started on the same one",
    )
    args = parser.parse_args(argv)
    if args.game_version != sts2_client.GAME_VERSION:
        parser.error(
            f"the host would run {sts2_client.GAME_VERSION}, the game is {args.game_version}: "
            f"set STS2_GAME_VERSION={args.game_version}"
        )

    # The Rust mirror is the search's model, not what is under test: a miss of
    # its own is logged and counted in `mirror_stats` instead of ending the run.
    os.environ.setdefault("STS2_MIRROR_OBSERVE", "1")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    from realgame_ppo import PpoPolicyCache

    args.out.mkdir(parents=True, exist_ok=True)
    results = args.out / "results.jsonl"
    done = set()
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["character"], row["ascension"], row["seed"]))
    ppo = PpoPolicyCache(None)
    meta = {
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "game_version": sts2_client.GAME_VERSION,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for character in args.characters.split(","):
        for ascension in (int(a) for a in args.ascensions.split(",")):
            for seed in range(args.seeds, args.seeds + args.runs):
                if (character, ascension, seed) in done:
                    continue
                logger.info("=== %s A%d seed %d ===", character, ascension, seed)
                try:
                    row = play_one(
                        character=character,
                        ascension=ascension,
                        seed=seed,
                        out_dir=args.out,
                        host=args.host,
                        port=args.port,
                        max_steps=args.max_steps,
                        max_run_seconds=args.max_run_seconds,
                        sims_regular=args.sims_regular,
                        sims_boss=args.sims_boss,
                        ppo_policies=ppo,
                    )
                except Exception as e:  # harness failure: record and move on
                    logger.exception("harness error")
                    row = RunRow(
                        character, ascension, seed, f"harness_error: {e}", 0, None, None, 0.0
                    )
                logger.info(
                    "%s A%d seed %d: %s at act %s floor %s after %d steps (%.0fs)%s",
                    character,
                    ascension,
                    seed,
                    row.result,
                    row.act,
                    row.floor,
                    row.steps,
                    row.duration_s,
                    f" -- {row.divergence['summary']}" if row.divergence else "",
                )
                with results.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(row), sort_keys=True) + "\n")
                if row.result not in ("victory", "defeat") and not args.no_restart:
                    restart_game(game_dir=args.game_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
