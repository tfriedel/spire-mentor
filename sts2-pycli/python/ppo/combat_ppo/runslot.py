"""A hybrid run slot: one full run in a rollout worker.

The run net (`ppo.net_policy.NetPolicy`, CPU, sampled at T = 1) makes every
multi-option non-combat decision, exactly as `workers.play_one(policy="net")`
does, and records the same `transitions`. Fights are not played here: the env
runs with `combat="yield"`, and each multi-option combat step becomes a
decision for the learner's inference server, like a curriculum fight's.

Each fight is its own combat-PPO episode with the curriculum's terminal reward
(a win is `1 + w_hp·hp_frac`, a loss `w_removed·removed`), so a fight inside a
run trains the combat net exactly as a fight from the start pool does. When the
run ends, its row (`play_one`'s format) goes to the run learner.

With `run_reward = "value"` a fight's reward is instead the change it makes to
the run critic's estimate of the run's outcome:
`run_reward_scale · (m_after − m_before)`, where `m(s) = V(s) + Φ(s)` is the run
net's value plus its shaping potential (`ppo.rewards.potential`), i.e. its
estimate of the terminal base value (graded floor progress + win bonus).
`m_before` is taken at the last run decision before the fight and `m_after` at
the first one after it. A fight that ends the run uses the terminal base value
instead (a lost fight: the death floor; the last boss won: the win).

A fight boundary is the phase leaving combat, the run ending, or the snapshot's
round counter going back (back-to-back fights with no non-combat step between
them), as in `agent.PolicyAgent`.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

import numpy as np

ROOMS = ("monster", "elite", "boss")


def fight_room(rm: Any) -> str:
    room = rm.current_room().get("room_type")
    return room if room in ROOMS else "monster"


class RunSlot:
    """One run at a time; `advance` plays it up to the next server decision."""

    def __init__(
        self,
        slot: int,
        seeds: Any,
        cfg: dict[str, Any],
        enc: Any,
        run_policy: Any,
        ends: list[dict[str, Any]],
        run_rows: list[dict[str, Any]],
    ) -> None:
        self.slot = slot
        self.seeds = seeds  # an iterator of run seeds
        self.cfg = cfg
        self.enc = enc
        self.policy = run_policy  # a callable returning the current NetPolicy
        self.ends = ends
        self.run_rows = run_rows
        self.max_steps = int(cfg["max_steps"])
        self.ep = -1  # the combat episode counter (one per fight)
        self.env: Any = None
        self.out: Any = None
        self.in_fight = False
        self.fight: dict[str, Any] = {}
        self.pending: list | None = None
        self.value_reward = cfg.get("run_reward", "fight") == "value"
        self.scale = float(cfg.get("run_reward_scale") or 1.0)
        self.shaping: Any = None
        if self.value_reward:
            from ppo.rewards import ShapingConfig

            self.shaping = ShapingConfig(**cfg["run_shaping"])
        self.new_run()

    # ── runs ─────────────────────────────────────────────────────────────

    def new_run(self) -> None:
        from ppo.env import PpoRunEnv

        self.seed = int(next(self.seeds))
        self.env = PpoRunEnv(
            seed=self.seed,
            ascension=int(self.cfg["run_ascension"]),
            character=self.cfg["run_character"],
            combat="yield",
            act1=self.cfg["run_act1"],
        )
        self.rng = np.random.default_rng(self.seed)
        self.transitions: list[dict[str, Any]] = []
        self.t0 = time.time()
        self.fights = 0
        self.fights_won = 0
        self.policy_ver = self.cfg["_run_ver"]()
        self.m_last: float | None = None  # the run value at the last run decision
        self.wait_m: list[dict[str, Any]] = []  # won fights waiting for `m_after`
        self.out = self.env.reset()
        self.in_fight = False

    def _end_run(self, terminal: Any, error: str | None) -> None:
        from ppo.workers import _reached_second_boss

        env = self.env
        if self.wait_m:
            if error:
                for rec in self.wait_m:
                    rec["error"] = error
                    self.ends.append(rec)
                self.wait_m = []
            else:
                self._resolve(self._terminal_m(terminal.metrics, bool(terminal.won)))
        row = {
            "seed": self.seed,
            "policy": "net",
            "ascension": env.ascension,
            "character": env.character,
            "act1": env.rm.act1,
            "outcome": "error" if error else terminal.outcome,
            "won": False if error else bool(terminal.won),
            "steps": env.steps,
            "n_decisions": env.decisions,
            "final": env.metrics_snapshot() if error else terminal.metrics,
            "reached_second_boss": None if error else _reached_second_boss(env.rm),
            "wall_s": round(time.time() - self.t0, 2),
            "fights": self.fights,
            "fights_won": self.fights_won,
            "run_ver": self.policy_ver,
            "error": error,
            "transitions": [] if error else self.transitions,
        }
        self.run_rows.append(row)
        self.new_run()

    # ── fights ───────────────────────────────────────────────────────────

    def _start_fight(self) -> None:
        self.ep += 1
        self.fights += 1
        self.in_fight = True
        self.fight = {
            "room": fight_room(self.env.rm),
            "encounter": str(self.env.rm.current_room().get("encounter") or ""),
            "ilog": self.enc.new_log(),
            "round": 0,
            "steps": 0,
            "decisions": 0,
            "removed": 0.0,
            "m0": self.m_last,
        }

    def _end_fight(self, won: bool, capped: bool, error: str | None) -> None:
        rm = self.env.rm
        f = self.fight
        hp_frac = 0.0
        if won:
            hp_frac = max(0.0, float(rm.current_hp) / max(1.0, float(rm.max_hp)))
            self.fights_won += 1
        rec = {
            "slot": self.slot,
            "ep": self.ep,
            "won": won,
            "hp_frac": hp_frac,
            "removed": f["removed"],
            "room": f["room"],
            "own": False,
            "run": True,
            "encounter": f["encounter"],
            "steps": f["steps"],
            "decisions": f["decisions"],
            "capped": capped,
            "error": error,
        }
        self.in_fight = False
        if not self.value_reward or error:
            self.ends.append(rec)
            return
        if f["m0"] is None:
            rec["error"] = "no run decision before the fight"
            self.ends.append(rec)
            return
        rec["m0"] = f["m0"]
        if not won or rm.is_over():
            m = self.env.metrics_snapshot()
            self.wait_m.append(rec)
            self._resolve(self._terminal_m(m, won and bool(rm.player_won)))
        else:
            self.wait_m.append(rec)

    def _terminal_m(self, metrics: dict[str, Any], won: bool) -> float:
        from ppo.rewards import base_value

        return base_value({"ascension": self.env.ascension, **metrics}, won, self.shaping)

    def _resolve(self, m1: float) -> None:
        """Reward the fights waiting for the run value after them."""
        for rec in self.wait_m:
            rec["m1"] = m1
            rec["r_run"] = self.scale * (m1 - rec["m0"])
            self.ends.append(rec)
        self.wait_m = []

    def _after_combat_action(self) -> None:
        """Close the fight if this action ended it."""
        rm = self.env.rm
        if rm.is_over():
            self._end_fight(bool(rm.player_won), capped=False, error=None)
        elif rm.phase() != "combat":
            self._end_fight(True, capped=False, error=None)

    # ── the step loop ────────────────────────────────────────────────────

    def advance(self) -> Any:
        """Play until a combat decision for the server; return it (a `Decision`).

        Run decisions, single-option combat steps and run ends are handled
        here. Engine errors abort the run (and its open fight).
        """
        from ppo.combat_bench.oracle import combat_to_rm
        from ppo.encoder import fine_surface
        from ppo.env import CombatTurn, Decision
        from ppo.workers import curse_removal

        while True:
            try:
                out = self.out
                if isinstance(out, Decision):
                    idx, logp, value, feat = self.policy().sample(
                        out.obs, out.actions, out.metrics, self.rng
                    )
                    if self.value_reward:
                        from ppo.rewards import potential

                        mt = {"ascension": self.env.ascension, **out.metrics}
                        self.m_last = value + potential(mt, self.shaping)
                        if self.wait_m:
                            self._resolve(self.m_last)
                    self.transitions.append(
                        {
                            "feat": feat,
                            "action_idx": idx,
                            "old_logprob": logp,
                            "value": value,
                            "metrics": out.metrics,
                            "surface": out.surface,
                            "fine_surface": fine_surface(out.actions),
                            "n_actions": len(out.actions),
                            "curse_rm": curse_removal(out.obs, out.actions, idx),
                            "bundle": bool(out.obs.get("card_reward_bundles")),
                        }
                    )
                    self.out = self.env.step(idx)
                    continue
                if not isinstance(out, CombatTurn):  # a Terminal
                    if self.in_fight:
                        rm = self.env.rm
                        self._end_fight(bool(rm.player_won), capped=False, error=None)
                    self._end_run(out, None)
                    continue
                rm = self.env.rm
                if rm.is_over() or rm.phase() != "combat":
                    if self.in_fight:
                        self._after_combat_action()
                    self.out = self.env.resume()
                    continue
                if not self.in_fight:
                    self._start_fight()
                f = self.fight
                cs = rm.combat_state()
                legal = cs.legal_actions()
                if not legal:
                    # The engine says combat but offers nothing: let the env
                    # decide (it ends the run as `no_actions` if so).
                    self._end_fight(False, capped=False, error="no legal combat action")
                    self.out = self.env.resume()
                    continue
                if f["steps"] >= self.max_steps or self.env.steps >= self.env.max_steps:
                    # A capped fight is lost, and so is the run.
                    self._end_fight(False, capped=True, error=None)
                    self._end_run(self.env._terminal("max_steps"), None)
                    continue
                snap, bv = self.enc.observe(f["ilog"], cs, f["room"])
                rnd = int(snap.get("round") or 0)
                if rnd < f["round"]:
                    # A new fight in the same combat phase (back-to-back bosses).
                    self._end_fight(True, capped=False, error=None)
                    self._start_fight()
                    f = self.fight
                    snap, bv = self.enc.observe(f["ilog"], cs, f["room"])
                    rnd = int(snap.get("round") or 0)
                f["round"] = rnd
                f["removed"] = _removed(cs, f["removed"])
                if len(legal) == 1:
                    rm.take_action(combat_to_rm(legal[0]))
                    self.env.combat_step()
                    f["steps"] += 1
                    self._after_combat_action()
                    continue
                d = self.enc.encode(snap, bv, cs, legal)
                self.pending = legal
                return d
            except BaseException as ex:  # a Rust panic is a BaseException
                if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                    raise
                err = traceback.format_exc(limit=4)
                if self.in_fight:
                    self._end_fight(False, capped=False, error=err)
                self._end_run(None, err)
                continue

    def apply(self, a: int) -> None:
        """Apply the server's action for the pending combat decision."""
        from ppo.combat_bench.oracle import combat_to_rm

        legal, self.pending = self.pending, None
        assert legal is not None
        try:
            self.env.rm.take_action(combat_to_rm(legal[int(a)]))
            self.env.combat_step()
            self.fight["steps"] += 1
            self.fight["decisions"] += 1
            self._after_combat_action()
        except BaseException as ex:
            if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                raise
            err = traceback.format_exc(limit=4)
            if self.in_fight:
                self._end_fight(False, capped=False, error=err)
            self._end_run(None, err)


def _removed(cs: Any, prev: float) -> float:
    """The fraction of enemy HP removed so far (`rollout_features` state[15])."""
    import sts2_rs

    try:
        return float(sts2_rs.rollout_features(cs, version=1)["state"][15])
    except Exception:
        return prev
