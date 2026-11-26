"""Lord's Parasol's forced card removal comes before the shop (GH #1908).

``LordsParasol.AfterRoomEntered`` starts ``PurchaseEverything`` fire-and-forget
(LordsParasol.cs:27). With interactive rewards (Spire Bench's referee) a bought
Orrery opens its reward screen in the middle of that buy-out (Orrery.cs:19-27),
and the answer to its last reward wakes the buy-out on another thread. The host
used to read the shop before that thread had bought the potions and reached the
non-cancelable removal (LordsParasol.cs:106-111): it reported a shop with the
potions still stocked and a paid removal on offer, and the forced removal
surfaced only on the client's next shop command (Spire Bench seeds 180000148,
180001197, 180001619). In the game the shop takes no input until the buy-out
ends, so the removal screen is the next decision.

Seed ``orrery3`` (Ironclad A0, v0.111.0) stocks Orrery in its shops once the
player owns Lord's Parasol; the first shop on the map is entered directly.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest
import sts2_client

pytestmark = pytest.mark.oracle

SEED = "orrery3"


class Host:
    def __init__(self) -> None:
        env = os.environ.copy()
        for var in ("STS2_REFEREE", "STS2_REFEREE_DIAG"):
            env.pop(var, None)
        env["STS2_INTERACTIVE_REWARDS"] = "1"
        self.proc = subprocess.Popen(
            sts2_client.headless_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
            cwd=sts2_client.REPO_ROOT,
        )
        self._read()

    def _read(self) -> dict:
        while True:
            line = self.proc.stdout.readline()
            assert line, "EOF from game process"
            if line.startswith("{"):
                return json.loads(line)

    def send(self, **cmd) -> dict:
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        return self._read()

    def act(self, action: str, **args) -> dict:
        return self.send(cmd="action", action=action, args=args)

    def __enter__(self) -> Host:
        return self

    def __exit__(self, *exc) -> None:
        self.proc.kill()
        self.proc.wait()


def _to_map(host: Host, state: dict) -> dict:
    for _ in range(20):
        decision = state.get("decision")
        if decision == "map_select":
            return state
        if decision == "event_choice":
            options = [o for o in state["options"] if not o.get("is_locked")]
            state = host.act("choose_option", option_index=options[0]["index"])
        elif decision == "card_reward":
            state = host.act("skip_card_reward")
        elif decision == "card_select" and state.get("min_select", 0) == 0:
            state = host.act("skip_select")
        elif decision == "card_select":
            state = host.act("select_cards", indices="0")
        else:
            state = host.act("proceed")
    return state


def test_the_forced_removal_follows_orrerys_last_reward() -> None:
    if sts2_client.GAME_VERSION != "v0.111.0":
        pytest.skip("seed orrery3's shop stock was found on v0.111.0")
    with Host() as host:
        state = _to_map(host, host.send(cmd="start_run", character="Ironclad", seed=SEED))
        assert state.get("decision") == "map_select", state.get("decision")
        r = host.send(
            cmd="set_player", hp=50, max_hp=80, gold=100, relics=["LORDS_PARASOL"], potions=[]
        )
        assert r.get("type") != "error", r
        rows = host.send(cmd="get_map")["rows"]
        shops = [(n["col"], n["row"]) for row in rows for n in row if n.get("type") == "Shop"]
        col, row = min(shops, key=lambda c: c[1])

        state = host.act("select_map_node", col=col, row=row)
        assert state.get("decision") == "rewards", "the shop no longer stocks Orrery"
        assert [r["type"] for r in state["rewards"]] == ["card"] * 5

        for _ in range(5):
            assert state.get("decision") == "rewards", state.get("decision")
            state = host.act("take_reward", reward_index=state["rewards"][0]["index"])
            assert state.get("decision") == "card_reward", state.get("decision")
            state = host.act("skip_card_reward")

        assert state.get("decision") == "card_select", (
            f"a {state.get('decision')} came before Lord's Parasol's forced removal"
        )
        assert state.get("source") == "FromDeckForRemoval"
        assert state.get("cancelable") is False
        assert len(state["player"]["potions"]) == 3, "the buy-out takes the potions first"

        state = host.act("select_cards", indices="0")
        assert state.get("decision") == "shop", state.get("decision")
        assert state.get("card_removal_cost") is None, "the removal is spent"
