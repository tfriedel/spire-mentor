"""`combat_bench` agent for a combat-PPO checkpoint: greedy, no search.

    python -m ppo.combat_bench run --set dev_v1 --agent ppo.combat_ppo.agent:make \
        --agent-arg ckpt=RUN/ckpt/final.pt ...

It plays exactly what `combat_ctx.play`'s `ctx:` policy plays: the honest
snapshot of `cs.belief_view(0)` plus this fight's own `IntentLog` (observed at
every agent step, forced ones included), the CPU fp32 `CombatCtxNet`, and the
argmax over `cs.legal_actions()`, mapped to the run action by
`combat_bench.oracle.combat_to_rm`. Outside combat it defers to the deployed
`MctsCombatAgent`. That never happens on a bench fight.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PolicyAgent:
    def __init__(self, seed: int, ckpt: str) -> None:
        import sts2_rs

        from ppo.combat_ctx.play import CtxNet

        self.policy = CtxNet(ckpt)
        self.fallback = sts2_rs.MctsCombatAgent(seed=int(seed))
        self._rm: Any = None
        self._fight: Any = None
        self._new_fight = True
        self._round = 0
        self._job: dict[str, Any] = {}
        self.decisions = 0
        self.fights = 0

    def combat_budget_spent(self) -> tuple[int, int, int, int]:
        return (self.decisions, 0, 0, 0)

    def step(self, rm: Any) -> dict[str, Any] | None:
        from ppo.combat_bench.oracle import combat_to_rm

        acts = rm.available_actions()
        if not acts:
            return None
        cs = rm.combat_state()
        if rm.phase() != "combat" or cs is None:
            return self.fallback.step(rm)
        room = (rm.act_index, rm.current_room().get("map_coord"))
        if rm is not self._rm or self._new_fight or room != self._fight:
            self._start_fight(rm)
        self._fight = room
        self.policy.observe(self._job, cs)
        rnd = int(self._job["snap"].get("round") or 0)
        if rnd < self._round:
            # The round counter went back: a new fight in the same room with no
            # combat-phase gap (back-to-back bosses). Restart its intent log.
            self._start_fight(rm)
            self.policy.observe(self._job, cs)
        self._round = rnd
        legal = cs.legal_actions()
        if len(legal) == 1:
            a = legal[0]
        else:
            z = self.policy.logits([(self._job, cs, legal)])[0]
            a = legal[int(np.argmax(z))]
            self.decisions += 1
        action = combat_to_rm(a)
        result = rm.take_action(action)
        if rm.is_over() or rm.phase() != "combat":
            self._new_fight = True
        return {
            "phase": "combat",
            "is_combat": True,
            "over": rm.is_over(),
            "won": bool(rm.player_won),
            "action": action,
            "combat_search": None,
            "result": result,
        }

    def _start_fight(self, rm: Any) -> None:
        """A fresh intent log for a new fight (`CtxNet.observe` makes one).

        In a bench fight each fight has its own `rm`. In a full run
        (`PpoRunEnv(combat="policy")`) one `rm` plays every fight, so the
        boundary is the combat phase ending, the room changing, or the round
        counter going back.
        """
        self._rm = rm
        self._new_fight = False
        self._round = 0
        self.fights += 1
        room = rm.current_room()["room_type"]
        # Event fights (and any other room) play as hallway fights: training
        # and the snapshot know only monster / elite / boss.
        self._job = {"row": {"room_type": room if room in ("elite", "boss") else "monster"}}


def make(seed: int, ckpt: str) -> PolicyAgent:
    """The `combat_bench run --agent ppo.combat_ppo.agent:make` factory."""
    return PolicyAgent(seed, ckpt)
