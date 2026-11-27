"""A ``sts2_client.Game``-shaped world that plays a Spire Bench episode.

``headless_eval.play_one_headless_session`` drives our search agent (PPO run
policy, MCTS fights with ``beliefdraw``) against anything that exposes
``start``/``send``/``act``/``get_map``/``proc``/``world_errors``. This class
answers those through the gateway, so the agent sees only the referee's
observation. What the agent's engine needs and the referee hides is
inferred, never read:

* every enemy's ``move_state`` and ``ai`` log come from :class:`MoveTracker`
  (the telegraphed intents, followed fight by fight);
* the act map is the last one the referee showed (``map_select``);
* ``game_version`` is the gateway's (``GET /v0/info``);
* draw-pile cards are numbered in the order the referee lists them (sorted
  by content, so the numbers say nothing about the draw order).

Two screens the headless host resolves itself are answered here the way the
host does, so the agent plays the decisions it was trained on: ``rewards``
takes every reward in turn (a card reward still opens ``card_reward`` for the
agent), and ``potion_swap`` declines (the host drops the potion).

No seed, RNG cursor or draw order exists here to read.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import crystal_sphere_policy
from spire_bench.client import SpireBench

from bench_agents.move_infer import MoveTracker


def host_answer(
    obs: dict[str, Any], seq: int, refused: set[tuple[int, str]]
) -> dict[str, Any] | None:
    """The answer to a screen the plain host settles itself, or None.

    Our decision loop has no head for these: the reward list (take each in
    order, skip what the game refuses, then proceed), a potion swap (decline)
    and the Crystal Sphere (the digger, ``crystal_sphere_policy``). The
    referee and the keyed control host both show them, so both arms answer
    them here, the same way. ``refused`` holds ``(seq, action json)`` the
    game turned down at this decision.
    """
    decision = obs.get("decision")
    if decision == "potion_swap":
        return {"action": "potion_swap"}
    if decision == "crystal_sphere":
        click = crystal_sphere_policy.host_click(obs)
        return None if click is None else {"action": "crystal_sphere_click", "args": click}
    if decision != "rewards":
        return None

    # C# lists every reward in `RewardsSetIndex` order (gold, potion, relic,
    # special card, card). The engine the student trained on takes a room's
    # relics before its potion (a full belt's swap comes after the relic),
    # an event fight's extra potion (PunchOff's prize) among them since
    # GH #1801. On a full belt, though, the engine asks that extra potion's
    # swap only after the card pick (GH #1832). Any order is the player's,
    # so the bench takes the engine's (GH #1698).
    player = obs.get("player") or {}
    slots = player.get("potion_slots")
    full = isinstance(slots, int) and len(player.get("potions") or []) >= slots

    def engine_rank(r: dict[str, Any]) -> int:
        kind = str(r.get("type"))
        if kind == "potion" and r.get("extra") and full:
            return 5
        return {"gold": 0, "relic": 1, "special_card": 2, "potion": 3, "card": 4}.get(kind, 6)

    rewards = sorted(obs.get("rewards") or [], key=engine_rank)
    for r in rewards:
        take = {"action": "take_reward", "args": {"reward_index": r["index"]}}
        if (seq, json.dumps(take, sort_keys=True)) in refused:
            skip = {"action": "skip_reward", "args": {"reward_index": r["index"]}}
            if (seq, json.dumps(skip, sort_keys=True)) not in refused:
                return skip
            continue
        return take
    return {"action": "proceed"}


class _NoProc:
    """``Game.proc`` stand-in: the process lives behind the gateway."""

    def kill(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        return 0


class RefereeGame:
    def __init__(self, bench: SpireBench) -> None:
        self.bench = bench
        self.game_version = bench.info()["game_version"]
        # Called with each screen the bench answers itself (GH #1704).
        self.on_auto: Callable[[dict[str, Any]], None] | None = None
        self.proc = _NoProc()
        self.episode_id: str | None = None
        self.seq = 0
        self.status = "running"
        self.fingerprint: str | None = None
        self.character = ""
        self.ascension = 0
        self._tracker: MoveTracker | None = None
        self._maps: dict[int, dict[str, Any]] = {}
        self._act: int | None = None
        self._refused: set[tuple[int, str]] = set()

    # ── Game interface ──────────────────────────────────────────────────────
    def start(self, character: str, *, seed: str = "", ascension: int = 0, **_: Any) -> dict:
        """Starts a referee episode; ``seed`` is ignored (the referee keys it)."""
        self.character, self.ascension = character, ascension
        (ep,) = self.bench.start(character, ascension)
        return self._take(ep)

    def send(self, cmd: dict[str, Any]) -> dict[str, Any]:
        if cmd.get("cmd") != "action":
            return {
                "type": "error",
                "message": f"not available through the referee: {cmd.get('cmd')}",
            }
        act = {"episode_id": self.episode_id, "seq": self.seq, "action": cmd["action"]}
        if cmd.get("args"):
            act["args"] = cmd["args"]
        (r,) = self.bench.step([act])
        if "error" in r and r.get("status") == "running":
            # Refused: the decision stands. Host-style error, like Game.send.
            return {"type": "error", "message": r["error"].get("code", "invalid")}
        return self._take(r)

    def act(self, action: str, **args: Any) -> dict[str, Any]:
        cmd: dict[str, Any] = {"cmd": "action", "action": action}
        if args := {k: v for k, v in args.items() if v is not None}:
            cmd["args"] = args
        return self.send(cmd)

    def get_map(self) -> dict[str, Any]:
        if self._act in self._maps:
            return self._maps[self._act]
        raise LookupError("the referee has not shown this act's map yet")

    def world_errors(self) -> dict[str, Any]:
        return {"count": 0, "sample": []}

    def close(self) -> None:
        pass

    # ── observation plumbing ────────────────────────────────────────────────
    def _take(self, view: dict[str, Any]) -> dict[str, Any]:
        while True:
            self.episode_id = view["episode_id"]
            self.seq = view["seq"]
            self.status = view["status"]
            self.fingerprint = view.get("fingerprint")
            obs = view.get("observation") or {}
            auto = self._host_answer(obs) if self.status == "running" else None
            if auto is not None and self.on_auto is not None:
                self.on_auto(obs)
            if auto is None:
                return self._enrich(dict(obs))
            act = {"episode_id": self.episode_id, "seq": self.seq, **auto}
            (view,) = self.bench.step([act])
            if "error" in view and view.get("status") == "running":
                self._refused.add((self.seq, json.dumps(auto, sort_keys=True)))

    def _host_answer(self, obs: dict[str, Any]) -> dict[str, Any] | None:
        return host_answer(obs, self.seq, self._refused)

    def _enrich(self, obs: dict[str, Any]) -> dict[str, Any]:
        obs["game_version"] = self.game_version
        ctx = obs.get("context") or {}
        if ctx.get("act") is not None:
            self._act = ctx["act"]
        elif obs.get("act") is not None:
            self._act = obs["act"]
        decision = obs.get("decision")
        for board in (obs, obs.get("combat")):
            if isinstance(board, dict) and isinstance(board.get("draw_pile"), list):
                board["draw_pile"] = [{**c, "index": i} for i, c in enumerate(board["draw_pile"])]
        if obs.get("map"):
            self._maps[self._act] = obs["map"]
        combat = obs if decision == "combat_play" else obs.get("combat")
        if isinstance(combat, dict) and combat.get("enemies"):
            if self._tracker is None:
                self._tracker = MoveTracker(self.character, self.ascension, self.game_version)
            combat.setdefault("game_version", self.game_version)
            alive, dead = self._tracker.infer_board(combat)
            for enemy, guess in [
                *zip(combat["enemies"], alive, strict=False),
                *zip(combat.get("dead_enemies") or [], dead, strict=False),
            ]:
                enemy["move_state"] = guess["move_state"]
                enemy["ai"] = guess["ai"]
        elif decision not in ("combat_play", "card_select", None):
            self._tracker = None  # out of the fight
        return obs
