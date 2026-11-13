"""Offline tests for :mod:`sts2_realgame` (real-game bridge phase 2).

Two tiers, mirroring ``test_realgame_adapter.py``:

* **Pure translation** -- call ``RealGame._translate_*`` directly on real
  captured bridge messages (``tests/fixtures/sts2_realgame/*.json``, copied
  verbatim from ``logs/realgame/20260711T*.jsonl`` -- see each fixture's
  ``_provenance``) and assert the result matches the headless
  ``Sts2Headless`` decision schema (``src/Sts2Headless/RunSimulator.cs``).
  No network involved.
* **Session round-trips** -- a :class:`FakeServer` (same idiom as
  ``test_realgame_bridge.py``) plays the SpireMentor side of the protocol on
  a background thread while a real :class:`~sts2_realgame.RealGame`
  instance drives the session in the foreground, proving the full
  decision-out / action-in loop (including the auto-resolved screens:
  ``treasure``, ``reward_screen``) never desyncs the blocking
  request/reply protocol.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from sts2_realgame import RealGame

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "sts2_realgame"
ADAPTER_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "realgame_adapter"


def _load(name: str) -> dict[str, Any]:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as fh:
        return json.load(fh)


def _state(name: str) -> dict[str, Any]:
    return _load(name)["state"]


# ──────────────────────────────────────────────────────────────────────────────
# FakeServer -- same idiom as test_realgame_bridge.py's helper, duplicated
# locally so this module stays self-contained (importing a sibling test
# module is fragile under pytest's collection).
# ──────────────────────────────────────────────────────────────────────────────


class FakeServer:
    """A single-connection TCP server under manual, test-driven control."""

    def __init__(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self.port = self._listener.getsockname()[1]
        self.conn: socket.socket | None = None
        self._accept_thread = threading.Thread(target=self._accept, daemon=True)
        self._accept_thread.start()

    def _accept(self) -> None:
        conn, _ = self._listener.accept()
        self.conn = conn

    def wait_for_connection(self, timeout: float = 5.0) -> socket.socket:
        deadline = time.monotonic() + timeout
        while self.conn is None:
            if time.monotonic() > deadline:
                raise TimeoutError("client never connected")
            time.sleep(0.01)
        return self.conn

    def send_json_line(self, obj: dict[str, Any]) -> None:
        assert self.conn is not None
        self.conn.sendall(json.dumps(obj).encode("utf-8") + b"\n")

    def recv_line(self, timeout: float = 5.0) -> dict[str, Any]:
        assert self.conn is not None
        self.conn.settimeout(timeout)
        buf = b""
        while b"\n" not in buf:
            chunk = self.conn.recv(8192)
            if not chunk:
                raise ConnectionError("client closed while server was reading")
            buf += chunk
        line, _, _ = buf.partition(b"\n")
        return json.loads(line.decode("utf-8"))

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
        self._listener.close()


@pytest.fixture
def fake_server() -> Iterator[FakeServer]:
    server = FakeServer()
    yield server
    server.close()


@pytest.fixture
def rg(fake_server: FakeServer) -> Iterator[RealGame]:
    game = RealGame("127.0.0.1", fake_server.port, reconnect_attempts=1, timeout=5.0)
    yield game
    game.close()


# ──────────────────────────────────────────────────────────────────────────────
# Tier 1: pure translation (no network) -- structural invariants against real
# captures.
# ──────────────────────────────────────────────────────────────────────────────


class TestTranslateMapSelect:
    def test_shape(self) -> None:
        game = RealGame.__new__(RealGame)
        state = _state("map_select.json")
        out = game._translate_map_select(state)

        assert out["type"] == "decision"
        assert out["decision"] == "map_select"
        assert out["act"] == state["act"]
        assert out["floor"] == state["floor"]
        bridge_nodes = state["nodes"]
        assert len(out["choices"]) == len(bridge_nodes)
        for bridge_node, choice in zip(bridge_nodes, out["choices"], strict=True):
            assert choice["col"] == bridge_node["col"]
            assert choice["row"] == bridge_node["row"]
            assert choice["type"] == bridge_node["type"]

    def test_caches_nodes_for_select_map_node(self) -> None:
        game = RealGame.__new__(RealGame)
        state = _state("map_select.json")
        game._translate_map_select(state)
        assert game._map_nodes == state["nodes"]


class TestTranslateRestSite:
    def test_shape(self) -> None:
        game = RealGame.__new__(RealGame)
        state = _state("rest_site.json")
        out = game._translate_rest_site(state)

        assert out["decision"] == "rest_site"
        bridge_opts = state["options"]
        assert len(out["options"]) == len(bridge_opts)
        for bridge_opt, opt in zip(bridge_opts, out["options"], strict=True):
            assert opt["index"] == bridge_opt["index"]
            assert opt["option_id"] == bridge_opt["id"]
            assert opt["name"] == bridge_opt["label"]
            assert opt["is_enabled"] == bridge_opt["enabled"]


class TestTranslateShop:
    def test_shape_and_index_mapping(self) -> None:
        game = RealGame.__new__(RealGame)
        state = _state("shop.json")
        out = game._translate_shop(state)

        assert out["decision"] == "shop"
        bridge_options = state["options"]
        bridge_cards = [o for o in bridge_options if o["action"] == "buy_card"]
        bridge_relics = [o for o in bridge_options if o["action"] == "buy_relic"]
        bridge_potions = [o for o in bridge_options if o["action"] == "buy_potion"]
        assert len(out["cards"]) == len(bridge_cards)
        assert len(out["relics"]) == len(bridge_relics)
        assert len(out["potions"]) == len(bridge_potions)
        assert out["card_removal_cost"] == state.get("purge_cost")

        # Headless buy_card(card_index) addresses `out["cards"]` positionally;
        # RealGame must remember which BRIDGE option index that maps back to.
        for i, bridge_card_opt in enumerate(bridge_cards):
            assert game._shop_map["cards"][i] == bridge_card_opt["index"]
            assert out["cards"][i]["name"] == bridge_card_opt["item"]["name"]
            assert out["cards"][i]["cost"] == bridge_card_opt["price"]
        for i, bridge_potion_opt in enumerate(bridge_potions):
            assert game._shop_map["potions"][i] == bridge_potion_opt["index"]

        remove_opt = next(o for o in bridge_options if o["action"] == "remove_card")
        assert game._shop_map["remove_card"] == remove_opt["index"]
        leave_opt = next(o for o in bridge_options if o["action"] == "leave_shop")
        assert game._shop_map["leave"] == leave_opt["index"]


class TestTranslateEvent:
    def test_shape(self) -> None:
        game = RealGame.__new__(RealGame)
        state = _state("event.json")
        out = game._translate_event(state)

        assert out["decision"] == "event_choice"
        bridge_opts = state["options"]
        assert len(out["options"]) == len(bridge_opts)
        for bridge_opt, opt in zip(bridge_opts, out["options"], strict=True):
            assert opt["index"] == bridge_opt["index"]
            assert opt["title"] == bridge_opt["label"]
            assert opt["description"] == bridge_opt["description"]
            assert opt["is_locked"] is (not bridge_opt["enabled"])
        assert out["event_name"] == bridge_opts[0]["event_id"]


class TestTranslateBossRelic:
    """boss_relic -> event_choice (never live-captured; see the fixture's
    ``_provenance`` -- this coverage is a best-effort structural check, not
    a verified-against-the-live-game one)."""

    def test_shape_matches_event_choice(self) -> None:
        game = RealGame.__new__(RealGame)
        state = _state("boss_relic.json")
        out = game._translate_boss_relic(state)

        assert out["decision"] == "event_choice"
        assert out["event_name"] == "Boss Relic"
        bridge_opts = state["options"]
        assert len(out["options"]) == len(bridge_opts)
        for bridge_opt, opt in zip(bridge_opts, out["options"], strict=True):
            assert opt["index"] == bridge_opt["index"]
            assert opt["title"] == bridge_opt["label"]
            assert opt["is_locked"] is False


class TestTranslateCardReward:
    def test_shape(self) -> None:
        game = RealGame.__new__(RealGame)
        game._gold_before_reward = None
        state = _state("card_reward.json")
        out = game._translate_card_reward(state)

        assert out["decision"] == "card_reward"
        assert out["can_skip"] is True
        bridge_cards = state["cards"]
        assert len(out["cards"]) == len(bridge_cards)
        for bridge_card, card in zip(bridge_cards, out["cards"], strict=True):
            assert card["index"] == bridge_card["index"]
            assert card["id"] == bridge_card["id"]
            assert card["rarity"] == bridge_card["rarity"]

    def test_gold_earned_uses_last_combat_snapshot(self) -> None:
        game = RealGame.__new__(RealGame)
        game._gold_before_reward = 82
        state = _state("card_reward.json")
        out = game._translate_card_reward(state)
        assert out["gold_earned"] == state["run_state"]["gold"] - 82

    def test_gold_earned_none_without_snapshot(self) -> None:
        game = RealGame.__new__(RealGame)
        game._gold_before_reward = None
        out = game._translate_card_reward(_state("card_reward.json"))
        assert out["gold_earned"] is None


class TestTranslateCardSelect:
    def test_shape(self) -> None:
        game = RealGame.__new__(RealGame)
        state = _state("card_select.json")
        out = game._translate_card_select(state)

        assert out["decision"] == "card_select"
        assert out["min_select"] == state["min_select"]
        assert out["max_select"] == state["max_select"]
        bridge_cards = state["cards"]
        assert len(out["cards"]) == len(bridge_cards)
        for bridge_card, card in zip(bridge_cards, out["cards"], strict=True):
            assert card["index"] == bridge_card["index"]
            assert card["id"] == bridge_card["id"]


class TestTranslateGameOver:
    def test_defeat(self) -> None:
        game = RealGame.__new__(RealGame)
        state = _state("game_over.json")
        out = game._translate_game_over(state)

        assert out["decision"] == "game_over"
        assert out["victory"] is (state["result"] == "victory")
        assert out["act"] == state["act"]
        assert out["floor"] == state["floor"]
        assert out["player"]["gold"] == state["run_state"]["gold"]
        assert out["player"]["deck_size"] == len(state["run_state"]["master_deck"])


class TestPlayerSummary:
    def test_relics_and_deck_translated(self) -> None:
        game = RealGame.__new__(RealGame)
        run_state = _state("shop.json")["run_state"]
        summary = game._player_summary(run_state)

        assert summary["gold"] == run_state["gold"]
        assert summary["hp"] == run_state["hp"]
        assert summary["max_hp"] == run_state["max_hp"]
        assert summary["block"] == 0
        assert len(summary["relics"]) == len(run_state["relics"])
        assert summary["relics"][0]["name"] == run_state["relics"][0]["name"]
        assert summary["deck_size"] == len(run_state["master_deck"])
        assert len(summary["deck"]) == len(run_state["master_deck"])

    def test_empty_potion_slots_dropped(self) -> None:
        game = RealGame.__new__(RealGame)
        run_state = _state("rest_site.json")["run_state"]
        summary = game._player_summary(run_state)
        assert all(not p.get("empty") for p in summary["potions"])
        assert len(summary["potions"]) == sum(1 for p in run_state["potions"] if not p.get("empty"))

    def test_potion_shape_matches_headless_no_slot_or_empty_keys(self) -> None:
        """Headless's ``PlayerSummary()`` potions are
        ``{index, name, description, vars, target_type}`` -- no ``slot``/
        ``empty`` keys (those are bridge-only bookkeeping, filtered out
        entirely, not merely nulled)."""
        game = RealGame.__new__(RealGame)
        run_state = _state("shop.json")["run_state"]
        summary = game._player_summary(run_state)
        for potion in summary["potions"]:
            assert set(potion.keys()) == {"index", "name", "description", "vars", "target_type"}
            assert "slot" not in potion
            assert "empty" not in potion

    def test_name_is_localized_title(self) -> None:
        """headless's ``name`` is ``"The Ironclad"``, not the bare
        character id/display name the bridge sends."""
        game = RealGame.__new__(RealGame)
        run_state = _state("shop.json")["run_state"]
        assert run_state["character"] == "IRONCLAD"
        summary = game._player_summary(run_state)
        assert summary["name"] == "The Ironclad"

    def test_name_falls_back_to_raw_id_when_unmapped(self) -> None:
        game = RealGame.__new__(RealGame)
        summary = game._player_summary({"character": "SOMETHING_NEW"})
        assert summary["name"] == "SOMETHING_NEW"

    def test_name_is_question_mark_when_character_missing(self) -> None:
        game = RealGame.__new__(RealGame)
        summary = game._player_summary({})
        assert summary["name"] == "?"

    def test_deck_card_id_has_card_prefix(self) -> None:
        """headless's deck cards emit ``c.Id.ToString()`` (``"CARD." +
        Entry``), not the bare id the bridge sends."""
        game = RealGame.__new__(RealGame)
        run_state = _state("shop.json")["run_state"]
        summary = game._player_summary(run_state)
        for bridge_card, deck_card in zip(run_state["master_deck"], summary["deck"], strict=True):
            assert deck_card["id"] == f"CARD.{bridge_card['id']}"


class TestTranslateCombat:
    """``combat_play``'s ``player``/``player_powers`` shape -- headless
    reuses the exact same ``PlayerSummary()`` for combat_play as every
    other decision (``RunSimulator.cs``), so gold/name/deck/relics/potions
    must all be present here too, not just on map_select/shop/etc."""

    def _combat_state(self, fixture_name: str) -> dict[str, Any]:
        return json.loads((ADAPTER_FIXTURE_DIR / fixture_name).read_text(encoding="utf-8"))["state"]

    def test_player_gold_present(self) -> None:
        game = RealGame.__new__(RealGame)
        game._gold_before_reward = None
        state = self._combat_state("single_enemy_ironclad.json")
        out = game._translate_combat(state)
        assert out["player"]["gold"] == state["run_state"]["gold"]

    def test_player_name_is_localized_title(self) -> None:
        game = RealGame.__new__(RealGame)
        game._gold_before_reward = None
        state = self._combat_state("single_enemy_ironclad.json")
        out = game._translate_combat(state)
        assert out["player"]["name"] == "The Ironclad"

    def test_player_potions_shape_matches_headless(self) -> None:
        game = RealGame.__new__(RealGame)
        game._gold_before_reward = None
        state = self._combat_state("single_enemy_ironclad.json")
        out = game._translate_combat(state)
        for potion in out["player"]["potions"]:
            assert set(potion.keys()) == {"index", "name", "description", "vars", "target_type"}

    def test_player_deck_present_with_card_prefixed_ids(self) -> None:
        game = RealGame.__new__(RealGame)
        game._gold_before_reward = None
        state = self._combat_state("single_enemy_ironclad.json")
        out = game._translate_combat(state)
        master_deck = state["run_state"]["master_deck"]
        assert out["player"]["deck_size"] == len(master_deck)
        assert len(out["player"]["deck"]) == len(master_deck)
        assert out["player"]["deck"][0]["id"] == f"CARD.{master_deck[0]['id']}"

    def test_player_relics_present_with_id_and_description(self) -> None:
        game = RealGame.__new__(RealGame)
        game._gold_before_reward = None
        state = self._combat_state("single_enemy_ironclad.json")
        out = game._translate_combat(state)
        bridge_relics = state["run_state"]["relics"]
        assert len(out["player"]["relics"]) == len(bridge_relics)
        assert out["player"]["relics"][0]["id"] == bridge_relics[0]["id"]
        assert out["player"]["relics"][0]["description"] == bridge_relics[0]["description"]

    def test_player_hp_block_untouched_from_live_combat_state(self) -> None:
        """hp/max_hp/block must stay the adapter's live per-turn combat
        values, not get clobbered by the (combat-blind) run_state ones."""
        game = RealGame.__new__(RealGame)
        game._gold_before_reward = None
        state = self._combat_state("single_enemy_ironclad.json")
        out = game._translate_combat(state)
        assert out["player"]["hp"] == state["player"]["hp"]
        assert out["player"]["max_hp"] == state["player"]["max_hp"]
        assert out["player"]["block"] == state["player"].get("block", 0)

    def test_player_powers_none_when_empty(self) -> None:
        """headless: ``player_powers`` is ``null`` (not ``[]``) when the
        player has no active powers."""
        game = RealGame.__new__(RealGame)
        game._gold_before_reward = None
        state = self._combat_state("single_enemy_ironclad.json")
        assert state["player"]["powers"] == []
        out = game._translate_combat(state)
        assert out["player_powers"] is None

    def test_player_powers_list_when_present(self) -> None:
        game = RealGame.__new__(RealGame)
        game._gold_before_reward = None
        state = self._combat_state("player_power_ironclad.json")
        bridge_powers = state["player"]["powers"]
        assert bridge_powers
        out = game._translate_combat(state)
        assert out["player_powers"] is not None
        assert len(out["player_powers"]) == len(bridge_powers)


# ──────────────────────────────────────────────────────────────────────────────
# Tier 2: full session round-trips against a FakeServer.
# ──────────────────────────────────────────────────────────────────────────────


def test_run_config_handshake(fake_server: FakeServer, rg: RealGame) -> None:
    """start() answers a run_config with start_run, then translates the
    first real decision it receives."""
    run_config = _state("run_config.json")
    map_select = _state("map_select.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start, character="Ironclad", seed=42, ascension=3)
        fake_server.wait_for_connection()
        fake_server.send_json_line(run_config)

        sent = fake_server.recv_line()
        assert sent["action"] == "start_run"
        assert sent["character"] == "IRONCLAD"
        assert sent["ascension"] == 3
        assert sent["seed"] == 42

        fake_server.send_json_line(map_select)
        result = future.result(timeout=5)

    assert result["decision"] == "map_select"


def test_run_config_off_returns_first_decision_directly(
    fake_server: FakeServer, rg: RealGame
) -> None:
    """When client_run_config is disabled, the mod's first message is
    already a real decision (no run_config handshake) -- start() must
    translate it directly instead of hanging waiting for one."""
    map_select = _state("map_select.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start, character="Ironclad")
        fake_server.wait_for_connection()
        fake_server.send_json_line(map_select)
        result = future.result(timeout=5)

    assert result["decision"] == "map_select"


def test_combat_play_round_trip(fake_server: FakeServer, rg: RealGame) -> None:
    """combat_action -> combat_play decision out; play_card -> bridge
    'play' verb in, with request_id/state_id echoed.

    Routed through a preceding map_select -> select_map_node hop (rather
    than feeding combat_state straight to ``_next_decision``) so the combat
    message passes through the normal ``client.receive_state()`` path and
    its ``request_id``/``state_id`` are actually recorded by the client for
    the echo assertion below.
    """
    combat_fixture = json.loads(
        (ADAPTER_FIXTURE_DIR / "single_enemy_ironclad.json").read_text(encoding="utf-8")
    )
    combat_state = combat_fixture["state"]
    map_select = _state("map_select.json")
    node = map_select["nodes"][0]

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(map_select)
        assert future.result(timeout=5)["decision"] == "map_select"

        future = pool.submit(rg.select_map_node, node["col"], node["row"])
        fake_server.recv_line()  # the select_map_node action itself
        fake_server.send_json_line(combat_state)
        decision = future.result(timeout=5)
        assert decision["decision"] == "combat_play"
        assert len(decision["hand"]) == len(combat_state["hand"])

        future = pool.submit(rg.play_card, 1, 0)
        sent = fake_server.recv_line()
        assert sent["action"] == "play"
        assert sent["card_index"] == 1
        assert sent["target_index"] == 0
        assert sent["request_id"] == combat_state["request_id"]
        assert sent["state_id"] == combat_state["state_id"]

        fake_server.send_json_line(_state("map_select.json"))
        result = future.result(timeout=5)

    assert result["decision"] == "map_select"


def test_finished_event_proceeds_without_surfacing(fake_server: FakeServer, rg: RealGame) -> None:
    """GH #1287: an event page whose only buttons leave the finished event
    (``is_proceed``) is clicked through, as the host walks to the map by
    itself once ``IsFinished``."""
    event = _state("event.json")
    finished = {
        **event,
        "options": [{**event["options"][0], "label": "Proceed", "is_proceed": True}],
    }

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(finished)

        sent = fake_server.recv_line()
        assert sent["action"] == "choose"
        assert sent["index"] == 0

        fake_server.send_json_line(_state("map_select.json"))
        result = future.result(timeout=5)

    assert result["decision"] == "map_select"


def test_event_with_a_choice_surfaces(fake_server: FakeServer, rg: RealGame) -> None:
    event = _state("event.json")
    event = {**event, "options": [{**o, "is_proceed": False} for o in event["options"]]}

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(event)
        result = future.result(timeout=5)

    assert result["decision"] == "event_choice"


def test_snapshot_notifications_are_skipped(fake_server: FakeServer, rg: RealGame) -> None:
    """GH #1287: the snapshot lane's fire-and-forget messages carry no decision."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line({"type": "snapshot_saved", "snapshot_id": "1-0", "seq": 0})
        fake_server.send_json_line(_state("map_select.json"))
        result = future.result(timeout=5)

    assert result["decision"] == "map_select"


def test_treasure_auto_resolves_without_surfacing(fake_server: FakeServer, rg: RealGame) -> None:
    """A treasure message must never reach the caller as a decision -- it
    is auto-answered with choose index 0 and the loop continues straight
    through to whatever comes next."""
    treasure = _state("treasure.json")
    map_select = _state("map_select.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(treasure)

        sent = fake_server.recv_line()
        assert sent == {
            "action": "choose",
            "index": 0,
            "request_id": treasure["request_id"],
            "state_id": treasure["state_id"],
        }

        fake_server.send_json_line(map_select)
        result = future.result(timeout=5)

    assert result["decision"] == "map_select"


def test_reward_screen_auto_collects_then_surfaces_card_reward(
    fake_server: FakeServer, rg: RealGame
) -> None:
    """reward_screen with gold + potion + card + proceed: RealGame must
    auto-collect the gold and potion buttons (never surfacing reward_screen
    itself) and then click the CardReward button, surfacing the resulting
    card_reward message as a real decision."""
    reward_screen = _state("reward_screen_with_card.json")
    card_reward = _state("card_reward.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(reward_screen)

        # First click: gold is index 0 in the fixture's own option list.
        sent = fake_server.recv_line()
        assert sent["action"] == "choose"
        assert sent["index"] == 0  # "gold:18"

        # Server re-sends reward_screen with gold's button gone and every
        # remaining option re-indexed from 0 (mirrors SimRewardsScreenHandler
        # re-numbering AvailableRewardButtons every iteration).
        shrunk = {
            **reward_screen,
            "options": [
                {**reward_screen["options"][1], "index": 0},  # PotionReward
                {**reward_screen["options"][2], "index": 1},  # CardReward
            ],
        }
        fake_server.send_json_line(shrunk)

        sent = fake_server.recv_line()
        assert sent["index"] == 0  # PotionReward, now at index 0

        shrunk_again = {
            **reward_screen,
            "options": [{**reward_screen["options"][2], "index": 0}],  # CardReward only
        }
        fake_server.send_json_line(shrunk_again)

        sent = fake_server.recv_line()
        assert sent["index"] == 0  # CardReward

        fake_server.send_json_line(card_reward)
        result = future.result(timeout=5)

    assert result["decision"] == "card_reward"


def test_reward_screen_gold_only_proceeds_to_next_room(
    fake_server: FakeServer, rg: RealGame
) -> None:
    """No CardReward at all: after collecting gold, RealGame clicks
    proceed and surfaces whatever comes after (not reward_screen)."""
    reward_screen = _state("reward_screen_gold_only.json")
    map_select = _state("map_select.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(reward_screen)

        sent = fake_server.recv_line()
        assert sent["index"] == 0  # gold

        shrunk = {**reward_screen, "options": [{**reward_screen["options"][1], "index": 0}]}
        fake_server.send_json_line(shrunk)

        sent = fake_server.recv_line()
        assert sent["index"] == 0  # proceed

        fake_server.send_json_line(map_select)
        result = future.result(timeout=5)

    assert result["decision"] == "map_select"


def _swap_reward_screen(base: dict[str, Any]) -> dict[str, Any]:
    """A reward screen with one full-belt potion reward (GH #1069): the mod's
    decline + one discard option per belt slot, then CardReward and proceed."""
    reward = {"id": "STABLE_SERUM", "name": "Stable Serum", "slot": -1}
    held = ["FIRE_POTION", "BLOCK_POTION", "ENERGY_POTION"]
    swaps = [
        {
            "index": 0,
            "id": "PotionReward",
            "action": "potion_swap",
            "label": "decline",
            "reward": reward,
            "reward_button": 0,
            "discard_slot": None,
        }
    ] + [
        {
            "index": 1 + k,
            "id": "PotionReward",
            "action": "potion_swap",
            "label": f"discard {pid}",
            "reward": reward,
            "reward_button": 0,
            "discard_slot": k,
        }
        for k, pid in enumerate(held)
    ]
    return {
        **base,
        "options": [
            *swaps,
            {"index": 4, "id": "CardReward", "action": "pick_reward"},
            {"index": 5, "id": "proceed", "action": "proceed"},
        ],
    }


def test_reward_screen_full_belt_potion_declined_by_default(
    fake_server: FakeServer, rg: RealGame
) -> None:
    """Without interactive_potion_swap the full-belt potion is left on the
    screen (its decline option), as the engines drop it with the swap off."""
    reward_screen = _swap_reward_screen(_state("reward_screen_with_card.json"))
    card_reward = _state("card_reward.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(reward_screen)

        sent = fake_server.recv_line()
        assert sent["index"] == 0  # decline

        fake_server.send_json_line(
            {
                **reward_screen,
                "options": [{"index": 0, "id": "CardReward", "action": "pick_reward"}],
            }
        )
        sent = fake_server.recv_line()
        assert sent["index"] == 0  # CardReward

        fake_server.send_json_line(card_reward)
        result = future.result(timeout=5)

    assert result["decision"] == "card_reward"


def test_reward_screen_full_belt_potion_surfaces_potion_swap(
    fake_server: FakeServer, rg: RealGame
) -> None:
    """interactive_potion_swap=True: the swap surfaces as the headless
    ``potion_swap`` decision, and ``act("potion_swap", discard_slot=1)``
    clicks the mod's option for slot 1."""
    reward_screen = _swap_reward_screen(_state("reward_screen_with_card.json"))
    card_reward = _state("card_reward.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start, interactive_potion_swap=True)
        fake_server.wait_for_connection()
        fake_server.send_json_line(reward_screen)
        decision = future.result(timeout=5)

        assert decision["decision"] == "potion_swap"
        assert decision["potion"] == "STABLE_SERUM"
        assert [o["discard_slot"] for o in decision["options"]] == [None, 0, 1, 2]
        assert [o["index"] for o in decision["options"]] == [0, 1, 2, 3]
        assert decision["options"][2]["label"] == "discard BLOCK_POTION"

        assert rg.act("potion_swap", discard_slot=7)["type"] == "error"

        future = pool.submit(rg.act, "potion_swap", discard_slot=1)
        sent = fake_server.recv_line()
        assert sent["action"] == "choose"
        assert sent["index"] == 2  # the mod's "discard slot 1" option

        fake_server.send_json_line(
            {
                **reward_screen,
                "options": [{"index": 0, "id": "CardReward", "action": "pick_reward"}],
            }
        )
        sent = fake_server.recv_line()
        assert sent["index"] == 0  # CardReward

        fake_server.send_json_line(card_reward)
        result = future.result(timeout=5)

    assert result["decision"] == "card_reward"


def test_boss_relic_round_trip_surfaces_as_event_choice(
    fake_server: FakeServer, rg: RealGame
) -> None:
    boss_relic = _state("boss_relic.json")
    map_select = _state("map_select.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(boss_relic)
        decision = future.result(timeout=5)
        assert decision["decision"] == "event_choice"
        assert decision["event_name"] == "Boss Relic"

        future = pool.submit(rg.choose_option, 1)
        sent = fake_server.recv_line()
        assert sent == {"action": "choose", "index": 1, "request_id": "999", "state_id": 999}

        fake_server.send_json_line(map_select)
        result = future.result(timeout=5)

    assert result["decision"] == "map_select"


def test_rest_site_choose_option_round_trip(fake_server: FakeServer, rg: RealGame) -> None:
    rest_site = _state("rest_site.json")
    map_select = _state("map_select.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(rest_site)
        decision = future.result(timeout=5)
        assert decision["decision"] == "rest_site"

        future = pool.submit(rg.choose_option, 1)  # "Smith"
        sent = fake_server.recv_line()
        assert sent["action"] == "choose"
        assert sent["index"] == 1

        fake_server.send_json_line(map_select)
        result = future.result(timeout=5)

    assert result["decision"] == "map_select"


def test_shop_buy_actions_use_bridge_index_mapping(fake_server: FakeServer, rg: RealGame) -> None:
    shop = _state("shop.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(shop)
        decision = future.result(timeout=5)
        assert decision["decision"] == "shop"

        # cards[2] ("Armaments") is bridge option index 3 in the fixture.
        future = pool.submit(rg.buy_card, 2)
        sent = fake_server.recv_line()
        assert sent["index"] == 3
        fake_server.send_json_line(shop)
        future.result(timeout=5)

        # potions[1] ("Energy Potion") is bridge option index 7.
        future = pool.submit(rg.buy_potion, 1)
        sent = fake_server.recv_line()
        assert sent["index"] == 7
        fake_server.send_json_line(shop)
        future.result(timeout=5)

        future = pool.submit(rg.remove_card)
        sent = fake_server.recv_line()
        assert sent["index"] == 9
        fake_server.send_json_line(shop)
        future.result(timeout=5)

        future = pool.submit(rg.leave_room)
        sent = fake_server.recv_line()
        assert sent["index"] == 0
        fake_server.send_json_line(_state("map_select.json"))
        result = future.result(timeout=5)

    assert result["decision"] == "map_select"


def test_select_map_node_resolves_col_row_to_bridge_index(
    fake_server: FakeServer, rg: RealGame
) -> None:
    map_select = _state("map_select.json")
    node = map_select["nodes"][0]

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(map_select)
        decision = future.result(timeout=5)
        assert decision["decision"] == "map_select"

        future = pool.submit(rg.select_map_node, node["col"], node["row"])
        sent = fake_server.recv_line()
        assert sent["index"] == node["index"]

        fake_server.send_json_line(_state("event.json"))
        result = future.result(timeout=5)

    assert result["decision"] == "event_choice"


def test_select_map_node_unknown_coord_errors_without_sending(
    fake_server: FakeServer, rg: RealGame
) -> None:
    map_select = _state("map_select.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(map_select)
        future.result(timeout=5)

    # No thread needed: an unreachable (col, row) is rejected client-side,
    # before anything is sent -- the server has nothing to answer.
    result = rg.select_map_node(99, 99)
    assert result["type"] == "error"


def test_game_over_then_run_complete_drained_and_cached(
    fake_server: FakeServer, rg: RealGame
) -> None:
    game_over = _state("game_over.json")
    run_complete = _state("run_complete.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(game_over)
        fake_server.send_json_line(run_complete)
        result = future.result(timeout=5)

    assert result["decision"] == "game_over"
    assert result["victory"] is False

    # Further actions must not touch the socket at all -- the FakeServer
    # has nothing more queued, so a stray send would time out recv_line
    # (or hang the test) if RealGame failed to short-circuit.
    again = rg.end_turn()
    assert again is result


def test_error_notification_is_transparent_and_keeps_ids_fresh(
    fake_server: FakeServer, rg: RealGame
) -> None:
    """A bridge `error` must never surface to the caller as the decision --
    it's swallowed, the guaranteed fresh re-send is drained, and its
    request_id/state_id become what the NEXT action echoes."""
    rest_site = _state("rest_site.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(rest_site)
        future.result(timeout=5)

        future = pool.submit(rg.choose_option, 5)  # deliberately out of range
        sent = fake_server.recv_line()
        assert sent["index"] == 5

        fake_server.send_json_line(
            {"type": "error", "error": "invalid_index", "detail": "index 5 out of range"}
        )
        fresh_resend = {**rest_site, "state_id": 9999, "request_id": "9999"}
        fake_server.send_json_line(fresh_resend)

        decision = future.result(timeout=5)
        # The error was swallowed -- the caller sees the re-sent decision,
        # not an {"type": "error"} dict.
        assert decision["decision"] == "rest_site"

        # The next action must echo the RESEND's ids, proving the client's
        # bookkeeping tracked the drained message, not the stale original.
        future = pool.submit(rg.choose_option, 0)
        sent = fake_server.recv_line()
        assert sent["request_id"] == "9999"
        assert sent["state_id"] == 9999

        fake_server.send_json_line(_state("map_select.json"))
        result = future.result(timeout=5)

    assert result["decision"] == "map_select"


def test_bundle_select_round_trip(fake_server: FakeServer, rg: RealGame) -> None:
    card_bundle = {
        "type": "card_bundle",
        "bundles": [
            {
                "index": 0,
                "action": "pick_card_bundle",
                "cards": [
                    {
                        "id": "STRIKE_IRONCLAD",
                        "name": "Strike",
                        "description": "Deal 6 damage.",
                        "cost": 1,
                        "effective_cost": 1,
                        "type": "Attack",
                    }
                ],
                "enabled": True,
            },
            {
                "index": 1,
                "action": "pick_card_bundle",
                "cards": [
                    {
                        "id": "DEFEND_IRONCLAD",
                        "name": "Defend",
                        "description": "Gain 5 Block.",
                        "cost": 1,
                        "effective_cost": 1,
                        "type": "Skill",
                    }
                ],
                "enabled": True,
            },
        ],
        "seed": "42",
        "act": 1,
        "floor": 0,
        "mod_version": "0.1.0",
        "run_state": _state("map_select.json")["run_state"],
        "state_id": 5,
        "request_id": "5",
    }

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(card_bundle)
        decision = future.result(timeout=5)
        assert decision["decision"] == "bundle_select"
        assert len(decision["bundles"]) == 2
        assert decision["bundles"][0]["cards"][0]["name"] == "Strike"

        future = pool.submit(rg.select_bundle, 1)
        sent = fake_server.recv_line()
        assert sent == {"action": "choose", "index": 1, "request_id": "5", "state_id": 5}

        fake_server.send_json_line(_state("map_select.json"))
        result = future.result(timeout=5)

    assert result["decision"] == "map_select"


def test_card_select_and_card_reward_action_translation(
    fake_server: FakeServer, rg: RealGame
) -> None:
    card_select = _state("card_select.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(card_select)
        decision = future.result(timeout=5)
        assert decision["decision"] == "card_select"

        future = pool.submit(rg.select_cards, "0,2")
        sent = fake_server.recv_line()
        assert sent["action"] == "choose"
        assert sent["indices"] == [0, 2]

        fake_server.send_json_line(_state("card_reward.json"))
        decision = future.result(timeout=5)
        assert decision["decision"] == "card_reward"

        future = pool.submit(rg.skip_card_reward)
        sent = fake_server.recv_line()
        assert sent["action"] == "skip"

        fake_server.send_json_line(_state("map_select.json"))
        result = future.result(timeout=5)

    assert result["decision"] == "map_select"


def test_potions_are_numbered_like_the_host_and_used_by_belt_slot() -> None:
    """GH #1287: the host numbers potions by position among the filled slots
    (`Player.Potions` skips empty ones); the bridge addresses belt slots."""
    rg = RealGame()
    summary = rg._player_summary(
        {
            "potions": [
                {"slot": 0, "empty": True},
                {"slot": 1, "name": "Blood Potion", "target_type": "AnyPlayer"},
            ]
        }
    )
    assert [p["index"] for p in summary["potions"]] == [0]
    assert rg._potion_slot(0) == 1


def test_a_skipped_card_reward_is_not_reopened(fake_server: FakeServer, rg: RealGame) -> None:
    """GH #1287: after a skip the reward stays on the rewards screen and the
    mod asks about the screen again; RealGame leaves instead of reopening it."""
    screen = _state("reward_screen_with_card.json")
    only_card = {
        **screen,
        "options": [o for o in screen["options"] if o["id"] in ("CardReward", "proceed")],
    }

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(_state("card_reward.json"))
        assert future.result(timeout=5)["decision"] == "card_reward"

        future = pool.submit(rg.skip_card_reward)
        assert fake_server.recv_line()["action"] == "skip"
        fake_server.send_json_line(only_card)
        sent = fake_server.recv_line()
        assert sent["index"] == next(
            o["index"] for o in only_card["options"] if o["id"] == "proceed"
        )
        fake_server.send_json_line(_state("map_select.json"))
        assert future.result(timeout=5)["decision"] == "map_select"


@pytest.mark.parametrize("copies", [1, 2])
def test_a_lone_ancient_node_is_entered_without_surfacing(
    fake_server: FakeServer, rg: RealGame, copies: int
) -> None:
    """GH #1287: headless enters the act-start Ancient by itself; the game
    offers it as the only map node, sometimes listed twice."""
    ancient = {
        **_state("map_select.json"),
        "nodes": [{"index": i, "type": "Ancient", "row": 0, "col": 3} for i in range(copies)],
    }

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rg.start)
        fake_server.wait_for_connection()
        fake_server.send_json_line(ancient)
        sent = fake_server.recv_line()
        assert sent["action"] == "choose"
        assert sent["index"] == 0
        fake_server.send_json_line(_state("event.json"))
        assert future.result(timeout=5)["decision"] == "event_choice"


def test_shop_stock_reports_a_used_removal_as_gone() -> None:
    """GH #1287: the stock row of a removal already bought this visit stays,
    unstocked and without an option; the host then sends no removal cost."""
    rg = RealGame("127.0.0.1", 1)  # never connects: translation only
    msg = {
        "type": "shop",
        "options": [{"index": 0, "action": "leave_shop"}],
        "stock": [
            {"kind": "buy_card", "label": "ANGER", "price": 50, "stocked": True, "option": 1},
            {"kind": "remove_card", "label": "remove_card", "price": 75, "stocked": False},
        ],
        "run_state": {"gold": 124},
    }
    shop = rg._translate_shop_stock(msg)
    assert shop["card_removal_cost"] is None
    assert rg.remove_card()["type"] == "error"
    msg["stock"][1].update(stocked=True, option=2)
    assert rg._translate_shop_stock(msg)["card_removal_cost"] == 75


def test_an_x_cost_card_costs_zero_as_the_host_says() -> None:
    """GH #1287: the bridge's ``effective_cost`` of an X card is the energy."""
    from sts2_realgame import _card_cost

    assert _card_cost({"cost": 0, "costs_x": True, "effective_cost": 3}) == 0
    assert _card_cost({"cost": 2, "effective_cost": 1}) == 1
    assert _card_cost({"cost": 2}) == 2
