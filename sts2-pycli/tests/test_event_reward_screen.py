"""Reward screens opened by events and relics are the player's (GH #1296).

``RewardsCmd.OfferCustom`` (Kaleidoscope's two card rewards, Neow's Bones'
relics, Toy Box's wax relics) shows the same NRewardsScreen as a fight: every
reward is optional and taken in any order. With interactive rewards the host
lists it as a ``rewards`` decision instead of taking it in list order.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest
import sts2_client

pytestmark = pytest.mark.oracle

# Neow offers Kaleidoscope on this seed (Defect A10, act 1 rolled) in both builds.
SEED = "PHVYRT2EHH6D"


class Host:
    def __init__(self, *, interactive: bool):
        env = os.environ.copy()
        for var in ("STS2_REFEREE", "STS2_REFEREE_DIAG", "STS2_INTERACTIVE_REWARDS"):
            env.pop(var, None)
        if interactive:
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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.proc.kill()
        self.proc.wait()


def _take_kaleidoscope(host: Host) -> dict:
    state = host.send(cmd="start_run", character="Defect", seed=SEED, ascension=10, act1="roll")
    option = next(
        o for o in state["options"] if str(o.get("text_key", "")).endswith(".KALEIDOSCOPE")
    )
    return host.act("choose_option", option_index=option["index"])


def test_kaleidoscopes_rewards_are_taken_in_the_players_order():
    with Host(interactive=True) as host:
        screen = _take_kaleidoscope(host)
        assert screen["decision"] == "rewards" and screen.get("from_event")
        assert [r["type"] for r in screen["rewards"]] == ["card", "card"]
        deck = screen["player"]["deck_size"]

        # The second reward first: its cards, not the first one's.
        second = host.act("take_reward", reward_index=1)
        assert second["decision"] == "card_reward" and len(second["cards"]) == 3
        back = host.act("select_card_reward", card_index=0)
        assert back["decision"] == "rewards"
        assert [r["index"] for r in back["rewards"]] == [0]
        assert back["player"]["deck_size"] == deck + 1

        # Leaving skips the first: no card from it.
        after = host.act("proceed")
        assert after["decision"] == "map_select"
        assert after["player"]["deck_size"] == deck + 1


def test_the_same_screen_gives_the_same_cards_in_either_order():
    """Taking order does not reroll anything: the offers were made on opening."""
    offers = []
    for first in (0, 1):
        with Host(interactive=True) as host:
            _take_kaleidoscope(host)
            shown = {}
            for index in (first, 1 - first):
                cards = host.act("take_reward", reward_index=index)["cards"]
                shown[index] = [c["id"] for c in cards]
                host.act("skip_card_reward")
            offers.append(shown)
    assert offers[0] == offers[1]


def test_a_default_host_still_offers_the_cards_one_after_another():
    with Host(interactive=False) as host:
        state = _take_kaleidoscope(host)
        assert state["decision"] == "card_reward"
