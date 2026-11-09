"""Offline tests for :mod:`realgame_bridge` and :mod:`realgame_random`.

No real game is involved: a fake TCP server is spun up on an ephemeral
localhost port in a background thread to exercise the wire protocol
(newline framing, split/batched messages, ``request_id`` echo,
reconnect-on-refused). The random policy functions are tested directly
against representative fixture state dicts drawn from PROTOCOL.md, with the
RNG seeded for determinism.
"""

from __future__ import annotations

import json
import random
import socket
import threading
import time
from collections.abc import Iterator

import pytest
from realgame_bridge import RealGameBridgeClient
from realgame_random import (
    choose_action,
    pick_card_bundle,
    pick_card_reward,
    pick_card_select,
    pick_combat_action,
    pick_map_select,
    pick_option_choice,
    pick_potion_action,
    pick_run_config,
)

# ----------------------------------------------------------------
# Fake server helpers
# ----------------------------------------------------------------


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

    def send_raw(self, data: bytes) -> None:
        assert self.conn is not None
        self.conn.sendall(data)

    def send_json_line(self, obj: dict) -> None:
        self.send_raw(json.dumps(obj).encode("utf-8") + b"\n")

    def recv_line(self, timeout: float = 5.0) -> dict:
        """Block for one newline-terminated JSON line from the client."""
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


def free_port() -> int:
    """Return a TCP port with nothing listening on it (for refusal tests)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ----------------------------------------------------------------
# Framing / protocol tests
# ----------------------------------------------------------------


def test_receive_state_reassembles_split_message(fake_server: FakeServer) -> None:
    client = RealGameBridgeClient(host="127.0.0.1", port=fake_server.port, reconnect_attempts=1)
    client.connect()
    fake_server.wait_for_connection()

    payload = json.dumps({"type": "combat_action", "request_id": "1", "hand": []}).encode("utf-8")
    # Split the single message across two sends with a short delay.
    fake_server.send_raw(payload[:10])
    time.sleep(0.05)
    fake_server.send_raw(payload[10:] + b"\n")

    state = client.receive_state()
    assert state["type"] == "combat_action"
    assert state["request_id"] == "1"
    client.disconnect()


def test_receive_state_splits_batched_messages(fake_server: FakeServer) -> None:
    client = RealGameBridgeClient(host="127.0.0.1", port=fake_server.port, reconnect_attempts=1)
    client.connect()
    fake_server.wait_for_connection()

    msg1 = json.dumps({"type": "map_select", "nodes": [], "request_id": "a"})
    msg2 = json.dumps({"type": "pong"})
    # Two full messages arrive in a single recv() chunk.
    fake_server.send_raw((msg1 + "\n" + msg2 + "\n").encode("utf-8"))

    first = client.receive_state()
    second = client.receive_state()
    assert first == {"type": "map_select", "nodes": [], "request_id": "a"}
    assert second == {"type": "pong"}
    client.disconnect()


def _advice_request(request_id: str, advice_for: str = "combat_action") -> dict:
    return {"type": "advice_request", "request_id": request_id, "advice_for": advice_for}


def _wait_buffered(client: RealGameBridgeClient, advice_for: str) -> bool:
    """`newer_advice_request_pending` once the server's bytes have landed."""
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if client.newer_advice_request_pending(advice_for):
            return True
        time.sleep(0.01)
    return False


def test_a_queued_advice_request_supersedes_the_one_received(fake_server: FakeServer) -> None:
    """#1208: two queued requests -> the first has a newer one behind it,
    the second does not; both still come out of receive_state in order."""
    client = RealGameBridgeClient(host="127.0.0.1", port=fake_server.port, reconnect_attempts=1)
    client.connect()
    fake_server.wait_for_connection()
    fake_server.send_json_line(_advice_request("1"))
    fake_server.send_json_line(_advice_request("2"))

    first = client.receive_state()
    assert first["request_id"] == "1"
    assert _wait_buffered(client, "combat_action")
    second = client.receive_state()
    assert second["request_id"] == "2"
    assert not client.newer_advice_request_pending("combat_action")
    client.disconnect()


def test_other_messages_do_not_supersede_a_request(fake_server: FakeServer) -> None:
    """A queued non-advice message, a request for another kind of advice (an
    AI partner's screen question behind its combat one) or a half-arrived
    line does not count; none of them is consumed."""
    client = RealGameBridgeClient(host="127.0.0.1", port=fake_server.port, reconnect_attempts=1)
    client.connect()
    fake_server.wait_for_connection()
    fake_server.send_json_line(_advice_request("1"))
    fake_server.send_json_line({"type": "human_action", "advice_for": "combat_action"})
    fake_server.send_json_line(_advice_request("2", advice_for="card_select"))
    partial = json.dumps(_advice_request("3")).encode("utf-8")
    fake_server.send_raw(partial[:20])

    assert client.receive_state()["request_id"] == "1"
    # Every queued byte is buffered once card_select's request is seen.
    assert _wait_buffered(client, "card_select")
    assert not client.newer_advice_request_pending("combat_action")
    fake_server.send_raw(partial[20:] + b"\n")
    assert _wait_buffered(client, "combat_action")
    assert client.receive_state()["type"] == "human_action"
    assert client.receive_state()["request_id"] == "2"
    assert client.receive_state()["request_id"] == "3"
    client.disconnect()


def test_nothing_queued_does_not_block(fake_server: FakeServer) -> None:
    client = RealGameBridgeClient(
        host="127.0.0.1", port=fake_server.port, reconnect_attempts=1, timeout=30.0
    )
    client.connect()
    fake_server.wait_for_connection()
    started = time.monotonic()
    assert not client.newer_advice_request_pending("combat_action")
    assert time.monotonic() - started < 1.0
    client.disconnect()


def test_send_action_echoes_last_request_id(fake_server: FakeServer) -> None:
    client = RealGameBridgeClient(host="127.0.0.1", port=fake_server.port, reconnect_attempts=1)
    client.connect()
    fake_server.wait_for_connection()

    fake_server.send_json_line({"type": "combat_action", "request_id": "req-42", "hand": []})
    client.receive_state()

    client.send_action({"action": "end_turn"})
    received = fake_server.recv_line()
    assert received == {"action": "end_turn", "request_id": "req-42"}
    client.disconnect()


def test_send_action_does_not_overwrite_explicit_request_id(fake_server: FakeServer) -> None:
    client = RealGameBridgeClient(host="127.0.0.1", port=fake_server.port, reconnect_attempts=1)
    client.connect()
    fake_server.wait_for_connection()

    fake_server.send_json_line({"type": "combat_action", "request_id": "req-42", "hand": []})
    client.receive_state()

    client.send_action({"action": "end_turn", "request_id": "explicit"})
    received = fake_server.recv_line()
    assert received["request_id"] == "explicit"
    client.disconnect()


def test_send_action_echoes_last_state_id(fake_server: FakeServer) -> None:
    client = RealGameBridgeClient(host="127.0.0.1", port=fake_server.port, reconnect_attempts=1)
    client.connect()
    fake_server.wait_for_connection()

    fake_server.send_json_line(
        {"type": "combat_action", "request_id": "req-1", "state_id": 7, "hand": []}
    )
    client.receive_state()

    client.send_action({"action": "end_turn"})
    received = fake_server.recv_line()
    assert received == {"action": "end_turn", "request_id": "req-1", "state_id": 7}
    client.disconnect()


def test_send_action_state_id_opt_out(fake_server: FakeServer) -> None:
    client = RealGameBridgeClient(host="127.0.0.1", port=fake_server.port, reconnect_attempts=1)
    client.connect()
    fake_server.wait_for_connection()

    fake_server.send_json_line(
        {"type": "combat_action", "request_id": "req-1", "state_id": 7, "hand": []}
    )
    client.receive_state()

    client.send_action({"action": "end_turn"}, echo_state_id=False)
    received = fake_server.recv_line()
    assert "state_id" not in received
    assert received["request_id"] == "req-1"
    client.disconnect()


def test_send_action_does_not_overwrite_explicit_state_id(fake_server: FakeServer) -> None:
    client = RealGameBridgeClient(host="127.0.0.1", port=fake_server.port, reconnect_attempts=1)
    client.connect()
    fake_server.wait_for_connection()

    fake_server.send_json_line(
        {"type": "combat_action", "request_id": "req-1", "state_id": 7, "hand": []}
    )
    client.receive_state()

    client.send_action({"action": "end_turn", "state_id": 999})
    received = fake_server.recv_line()
    assert received["state_id"] == 999
    client.disconnect()


def test_error_message_flows_through_and_sets_last_error(fake_server: FakeServer) -> None:
    """An `error` notification must not desync the request/reply loop: the
    mod always follows it with a fresh copy of the decision it rejected."""
    client = RealGameBridgeClient(host="127.0.0.1", port=fake_server.port, reconnect_attempts=1)
    client.connect()
    fake_server.wait_for_connection()

    fake_server.send_json_line(
        {"type": "combat_action", "request_id": "req-1", "state_id": 1, "hand": []}
    )
    client.receive_state()
    assert client.last_error is None

    fake_server.send_json_line(
        {
            "type": "error",
            "error": "invalid_target",
            "detail": "target_index 9 out of range",
        }
    )
    error_state = client.receive_state()
    assert error_state["type"] == "error"
    assert error_state["error"] == "invalid_target"
    assert client.last_error == error_state

    # Mod re-sends a fresh copy of the decision (new request_id/state_id) --
    # the loop must survive and keep working normally.
    fake_server.send_json_line(
        {"type": "combat_action", "request_id": "req-2", "state_id": 2, "hand": []}
    )
    resent = client.receive_state()
    assert resent["type"] == "combat_action"
    assert client.last_error is None  # cleared by the non-error message

    client.send_action({"action": "end_turn"})
    received = fake_server.recv_line()
    assert received == {"action": "end_turn", "request_id": "req-2", "state_id": 2}
    client.disconnect()


def test_start_run_sends_expected_payload_and_echoes_ids(fake_server: FakeServer) -> None:
    client = RealGameBridgeClient(host="127.0.0.1", port=fake_server.port, reconnect_attempts=1)
    client.connect()
    fake_server.wait_for_connection()

    fake_server.send_json_line(
        {
            "type": "run_config",
            "request_id": "rc-1",
            "state_id": 1,
            "characters": [{"id": "IRONCLAD", "name": "The Ironclad"}],
            "max_ascension": 10,
            "defaults": {"character": "IRONCLAD", "ascension": 0, "seed": None},
        }
    )
    state = client.receive_state()
    assert state["type"] == "run_config"

    client.start_run("IRONCLAD", ascension=5, seed=1234)
    received = fake_server.recv_line()
    assert received == {
        "action": "start_run",
        "character": "IRONCLAD",
        "ascension": 5,
        "seed": 1234,
        "request_id": "rc-1",
        "state_id": 1,
    }
    client.disconnect()


def test_receive_state_ignores_unknown_fields(fake_server: FakeServer) -> None:
    """Unknown extra fields (seed/floor/act/mod_version) must not break parsing."""
    client = RealGameBridgeClient(host="127.0.0.1", port=fake_server.port, reconnect_attempts=1)
    client.connect()
    fake_server.wait_for_connection()

    fake_server.send_json_line(
        {
            "type": "combat_action",
            "request_id": "1",
            "hand": [],
            "enemies": [],
            "seed": "ABC123",
            "floor": 5,
            "act": 1,
            "mod_version": "0.1.0",
        }
    )
    state = client.receive_state()
    assert state["seed"] == "ABC123"
    assert state["type"] == "combat_action"
    client.disconnect()


def test_context_manager_connects_and_disconnects(fake_server: FakeServer) -> None:
    with RealGameBridgeClient(
        host="127.0.0.1", port=fake_server.port, reconnect_attempts=1
    ) as client:
        fake_server.wait_for_connection()
        assert client.connected
    assert not client.connected


def test_receive_state_raises_on_disconnect(fake_server: FakeServer) -> None:
    client = RealGameBridgeClient(host="127.0.0.1", port=fake_server.port, reconnect_attempts=1)
    client.connect()
    conn = fake_server.wait_for_connection()
    conn.close()

    with pytest.raises(ConnectionError):
        client.receive_state()
    client.disconnect()


def test_connect_reconnects_bounded_retries_then_raises() -> None:
    port = free_port()  # nothing listening
    client = RealGameBridgeClient(
        host="127.0.0.1", port=port, reconnect_attempts=3, reconnect_delay=0.01, timeout=1.0
    )
    start = time.monotonic()
    with pytest.raises(ConnectionError, match="after 3 attempts"):
        client.connect()
    elapsed = time.monotonic() - start
    # 3 attempts with a 0.01s delay between them should be near-instant;
    # bound generously to catch an accidental unbounded retry loop.
    assert elapsed < 5.0


def test_connect_succeeds_after_server_starts_late() -> None:
    """Reconnect loop should pick up a server that starts after the first attempt."""
    port = free_port()
    client = RealGameBridgeClient(
        host="127.0.0.1", port=port, reconnect_attempts=20, reconnect_delay=0.05, timeout=2.0
    )

    listener_holder: dict[str, socket.socket] = {}

    def start_late() -> None:
        time.sleep(0.15)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
        listener_holder["listener"] = listener
        listener.accept()

    t = threading.Thread(target=start_late, daemon=True)
    t.start()
    try:
        client.connect()
        assert client.connected
    finally:
        client.disconnect()
        listener = listener_holder.get("listener")
        if listener is not None:
            listener.close()


# ----------------------------------------------------------------
# Random policy legality tests
# ----------------------------------------------------------------

COMBAT_STATE = {
    "type": "combat_action",
    "request_id": "1",
    "state_id": 100,
    "player": {"hp": 50, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3},
    "hand": [
        {
            "id": "STRIKE_IRONCLAD",
            "cost": 1,
            "type": "Attack",
            "target_type": "AnyEnemy",
            "playable": True,
            "valid_target_indices": [1],
        },
        {
            "id": "DEFEND_IRONCLAD",
            "cost": 1,
            "type": "Skill",
            "target_type": "Self",
            "playable": True,
        },
        {
            "id": "BASH",
            "cost": 2,
            "type": "Attack",
            "target_type": "AnyEnemy",
            "playable": False,
            "valid_target_indices": [1],
        },
        {
            "id": "CLEAVE",
            "cost": 1,
            "type": "Attack",
            "target_type": "AllEnemies",
            "playable": True,
        },
    ],
    "enemies": [
        {
            "id": "JAW_WORM",
            "hp": 0,
            "max_hp": 44,
            "block": 0,
            "is_alive": False,
            "intent": "SingleAttack",
        },
        {
            "id": "JAW_WORM",
            "hp": 20,
            "max_hp": 40,
            "block": 0,
            "is_alive": True,
            "intent": "Defend",
        },
    ],
    "draw_pile_count": 5,
    "discard_pile_count": 2,
    "exhaust_pile_count": 0,
    "round": 2,
    "floor": 5,
    "act": 1,
}


def _assert_legal_combat_action(state: dict, action: dict) -> None:
    hand = state.get("hand") or []
    enemies = state.get("enemies") or []
    if action["action"] == "end_turn":
        return
    assert action["action"] == "play"
    idx = action["card_index"]
    assert 0 <= idx < len(hand)
    assert hand[idx]["playable"] is True
    target = action.get("target_index")
    card = hand[idx]
    if card.get("target_type") == "AnyEnemy":
        assert target is not None and 0 <= target < len(enemies)
        assert enemies[target].get("is_alive", True) is True
        valid = card.get("valid_target_indices")
        if valid is not None:
            assert target in valid
    else:
        assert target == -1


def test_pick_combat_action_always_legal_across_seeds() -> None:
    for seed in range(200):
        rng = random.Random(seed)
        action = pick_combat_action(COMBAT_STATE, rng)
        _assert_legal_combat_action(COMBAT_STATE, action)


def test_pick_combat_action_never_plays_unplayable_card() -> None:
    for seed in range(200):
        rng = random.Random(seed)
        action = pick_combat_action(COMBAT_STATE, rng)
        if action["action"] == "play":
            assert action["card_index"] != 2  # BASH is not playable


def test_pick_combat_action_no_playable_cards_ends_turn() -> None:
    state = dict(COMBAT_STATE)
    state["hand"] = [dict(c, playable=False) for c in COMBAT_STATE["hand"]]
    rng = random.Random(0)
    assert pick_combat_action(state, rng) == {"action": "end_turn"}


def test_pick_combat_action_skips_any_enemy_cards_with_no_alive_enemies() -> None:
    """If every enemy is dead, an AnyEnemy-target card must never be chosen.

    Mirrors what a real server would send: with no alive enemy,
    ``valid_target_indices`` on every AnyEnemy card comes back empty too
    (it's computed from ``CardModel.IsValidTarget``, which already accounts
    for aliveness) — not just ``enemies[].is_alive``.
    """
    state = dict(COMBAT_STATE)
    state["enemies"] = [dict(e, is_alive=False) for e in COMBAT_STATE["enemies"]]
    state["hand"] = [
        dict(c, valid_target_indices=[]) if c.get("target_type") == "AnyEnemy" else c
        for c in COMBAT_STATE["hand"]
    ]
    for seed in range(200):
        rng = random.Random(seed)
        action = pick_combat_action(state, rng)
        if action["action"] == "play":
            card = state["hand"][action["card_index"]]
            assert card.get("target_type") != "AnyEnemy"


def test_pick_combat_action_uses_valid_target_indices_over_raw_alive() -> None:
    """A card's `valid_target_indices` can be narrower than "any alive enemy"
    (e.g. an untargetable-while-stealthed effect) — the policy must respect
    that narrower set, never falling back to a wider "any alive" guess."""
    state = dict(COMBAT_STATE)
    # Two alive enemies, but STRIKE_IRONCLAD's valid_target_indices only
    # allows index 1 (simulating enemy 0 being untargetable for some reason
    # despite being alive).
    state["enemies"] = [
        dict(COMBAT_STATE["enemies"][0], is_alive=True),
        COMBAT_STATE["enemies"][1],
    ]
    state["hand"] = [
        dict(COMBAT_STATE["hand"][0], valid_target_indices=[1]),
        *COMBAT_STATE["hand"][1:],
    ]
    for seed in range(200):
        rng = random.Random(seed)
        action = pick_combat_action(state, rng)
        if action["action"] == "play" and action["card_index"] == 0:
            assert action["target_index"] == 1


MAP_STATE = {
    "type": "map_select",
    "nodes": [
        {"index": 0, "type": "Monster", "row": 2, "col": 1},
        {"index": 1, "type": "Elite", "row": 2, "col": 3},
        {"index": 2, "type": "Unknown", "row": 2, "col": 5},
    ],
    "floor": 3,
    "act": 1,
}


def test_pick_map_select_index_in_range() -> None:
    for seed in range(50):
        rng = random.Random(seed)
        action = pick_map_select(MAP_STATE, rng)
        assert action["action"] == "choose"
        assert 0 <= action["index"] < len(MAP_STATE["nodes"])


CARD_REWARD_STATE = {
    "type": "card_reward",
    "cards": [
        {"id": "INFLAME", "type": "Power", "cost": 1},
        {"id": "CARNAGE", "type": "Attack", "cost": 2},
        {"id": "SHRUG_IT_OFF", "type": "Skill", "cost": 1},
    ],
    "can_skip": True,
}


def test_pick_card_reward_respects_can_skip() -> None:
    saw_skip = False
    saw_pick = False
    for seed in range(200):
        rng = random.Random(seed)
        action = pick_card_reward(CARD_REWARD_STATE, rng)
        if action["action"] == "skip":
            saw_skip = True
        else:
            saw_pick = True
            assert action["action"] == "choose"
            assert 0 <= action["index"] < len(CARD_REWARD_STATE["cards"])
    assert saw_skip and saw_pick


def test_pick_card_reward_cannot_skip_always_picks() -> None:
    state = dict(CARD_REWARD_STATE, can_skip=False)
    for seed in range(50):
        rng = random.Random(seed)
        action = pick_card_reward(state, rng)
        assert action["action"] == "choose"
        assert 0 <= action["index"] < len(state["cards"])


EVENT_STATE = {
    "type": "event",
    "options": [
        {"index": 0, "action": "fight", "label": "Fight", "enabled": True},
        {"index": 1, "action": "leave", "label": "Leave", "enabled": False},
        {"index": 2, "action": "pay_gold", "label": "Pay 50 gold", "enabled": True},
    ],
    "floor": 8,
    "act": 1,
}


def test_pick_option_choice_only_picks_enabled(fake_server: FakeServer) -> None:
    del fake_server  # unused, keeps signature symmetric with other tests
    for seed in range(200):
        rng = random.Random(seed)
        action = pick_option_choice(EVENT_STATE, rng)
        assert action["action"] == "choose"
        idx = action["index"]
        assert EVENT_STATE["options"][idx]["enabled"] is True


def test_pick_option_choice_falls_back_to_skip_when_none_enabled() -> None:
    state = dict(EVENT_STATE, options=[dict(o, enabled=False) for o in EVENT_STATE["options"]])
    rng = random.Random(0)
    assert pick_option_choice(state, rng) == {"action": "skip"}


def test_pick_option_choice_withholds_world_flagged_lethal_options() -> None:
    """``will_kill_player`` is honoured by the random-legal fallback too.

    The shipped game's own random auto-player does this
    (``AutoSlay.EventRoomHandler``), so the fallback should not be the one path
    that still walks into ``Trial.DoubleDown``.
    """
    enabled = [i for i, o in enumerate(EVENT_STATE["options"]) if o.get("enabled", True)]
    assert len(enabled) >= 2, "fixture must offer a real choice for this to mean anything"
    lethal = enabled[0]
    options = [
        dict(o, will_kill_player=True) if i == lethal else dict(o)
        for i, o in enumerate(EVENT_STATE["options"])
    ]
    state = dict(EVENT_STATE, options=options)
    for seed in range(200):
        assert pick_option_choice(state, random.Random(seed))["index"] != lethal


def test_pick_option_choice_all_lethal_still_answers() -> None:
    # Same "unless every option is" rule as AutoSlay: never return `skip`
    # just because every remaining option is flagged.
    options = [
        dict(o, will_kill_player=True) if o.get("enabled", True) else dict(o)
        for o in EVENT_STATE["options"]
    ]
    state = dict(EVENT_STATE, options=options)
    action = pick_option_choice(state, random.Random(0))
    assert action["action"] == "choose"
    assert state["options"][action["index"]].get("enabled", True) is True


CARD_SELECT_STATE = {
    "type": "card_select",
    "cards": [
        {"id": "STRIKE_IRONCLAD", "type": "Attack", "cost": 1},
        {"id": "DEFEND_IRONCLAD", "type": "Skill", "cost": 1},
        {"id": "BASH", "type": "Attack", "cost": 2},
    ],
    # min_count/max_count (phase 1b) are additive aliases of min_select/
    # max_select; keep both here so this fixture also exercises the legacy
    # spelling still working via the fallback in pick_card_select.
    "min_select": 1,
    "max_select": 2,
    "min_count": 1,
    "max_count": 2,
    "cancelable": False,
}


def test_pick_card_select_respects_min_max() -> None:
    for seed in range(200):
        rng = random.Random(seed)
        action = pick_card_select(CARD_SELECT_STATE, rng)
        assert action["action"] == "choose"
        indices = action["indices"]
        assert 1 <= len(indices) <= 2
        assert len(set(indices)) == len(indices)
        assert all(0 <= i < 3 for i in indices)


def test_pick_card_select_single_pick_without_bounds() -> None:
    state = {"type": "card_select", "cards": CARD_SELECT_STATE["cards"]}
    for seed in range(50):
        rng = random.Random(seed)
        action = pick_card_select(state, rng)
        assert action["action"] == "choose"
        assert 0 <= action["index"] < 3


def test_pick_card_select_legacy_min_select_only_still_works() -> None:
    """Pre-phase-1b servers only send min_select/max_select -- must still work."""
    state = {
        "type": "card_select",
        "cards": CARD_SELECT_STATE["cards"],
        "min_select": 2,
        "max_select": 2,
    }
    for seed in range(50):
        rng = random.Random(seed)
        action = pick_card_select(state, rng)
        assert action["action"] == "choose"
        assert len(action["indices"]) == 2


def test_pick_card_select_never_cancels_when_not_cancelable() -> None:
    state = dict(CARD_SELECT_STATE, cancelable=False)
    for seed in range(200):
        rng = random.Random(seed)
        action = pick_card_select(state, rng)
        assert action["action"] != "cancel"


def test_pick_card_select_sometimes_cancels_when_cancelable() -> None:
    state = dict(CARD_SELECT_STATE, cancelable=True, min_count=0, min_select=0)
    saw_cancel = False
    saw_choice = False
    for seed in range(500):
        rng = random.Random(seed)
        action = pick_card_select(state, rng)
        if action["action"] in ("cancel", "skip"):
            saw_cancel = True
            assert action == {"action": action["action"]}
        else:
            saw_choice = True
            assert action["action"] == "choose"
    assert saw_cancel and saw_choice


CARD_BUNDLE_STATE = {
    "type": "card_bundle",
    "bundles": [
        {"index": 0, "action": "pick_card_bundle", "cards": [], "enabled": False},
        {"index": 1, "action": "pick_card_bundle", "cards": [], "enabled": True},
        {"index": 2, "action": "pick_card_bundle", "cards": [], "enabled": True},
    ],
    "floor": 1,
    "act": 1,
}


def test_pick_card_bundle_only_picks_enabled() -> None:
    for seed in range(200):
        rng = random.Random(seed)
        action = pick_card_bundle(CARD_BUNDLE_STATE, rng)
        assert action["action"] == "choose"
        assert CARD_BUNDLE_STATE["bundles"][action["index"]]["enabled"] is True


# ----------------------------------------------------------------
# Potion policy (phase 1b)
# ----------------------------------------------------------------


def test_pick_potion_action_targeted_shape_and_target_legality() -> None:
    potion = {"slot": 2, "requires_target": True}
    alive = [1, 3]
    saw_use = False
    saw_discard = False
    for seed in range(300):
        rng = random.Random(seed)
        action = pick_potion_action(potion, alive, rng)
        assert action is not None  # alive is non-empty, must never bail
        assert action["action"] == "potion"
        assert action["slot"] == 2
        if action.get("discard"):
            saw_discard = True
            assert "target_index" not in action
        else:
            saw_use = True
            assert action["target_index"] in alive
    assert saw_use and saw_discard


def test_pick_potion_action_self_target_has_no_target_index() -> None:
    potion = {"slot": 1, "requires_target": False}
    for seed in range(100):
        rng = random.Random(seed)
        action = pick_potion_action(potion, [], rng)
        assert action is not None
        assert action["slot"] == 1
        if not action.get("discard"):
            assert "target_index" not in action


def test_pick_potion_action_returns_none_when_target_required_but_none_alive() -> None:
    potion = {"slot": 0, "requires_target": True}
    saw_none = False
    for seed in range(300):
        rng = random.Random(seed)
        action = pick_potion_action(potion, [], rng)
        if action is None:
            saw_none = True
        else:
            # The only legal potion action with no alive target is a discard.
            assert action.get("discard") is True
    assert saw_none


POTION_COMBAT_STATE = {
    **COMBAT_STATE,
    "potions": [
        {
            "slot": 0,
            "empty": False,
            "id": "FIRE_POTION",
            "can_use": True,
            "target_type": "AnyEnemy",
            "requires_target": True,
        },
        {"slot": 1, "empty": True},
        {
            "slot": 2,
            "empty": False,
            "id": "BLOCK_POTION",
            "can_use": False,  # e.g. Automatic usage -- never a legal potion action
            "target_type": "Self",
            "requires_target": False,
        },
    ],
}


def test_pick_combat_action_sometimes_uses_the_one_usable_potion() -> None:
    saw_potion = False
    saw_other = False
    for seed in range(500):
        rng = random.Random(seed)
        action = pick_combat_action(POTION_COMBAT_STATE, rng)
        if action["action"] == "potion":
            saw_potion = True
            assert action["slot"] == 0  # only usable, non-empty slot
        else:
            saw_other = True
    assert saw_potion and saw_other


def test_pick_combat_action_never_targets_empty_or_unusable_potion_slot() -> None:
    for seed in range(500):
        rng = random.Random(seed)
        action = pick_combat_action(POTION_COMBAT_STATE, rng)
        if action["action"] == "potion":
            assert action["slot"] not in (1, 2)


def test_pick_combat_action_no_potions_key_behaves_as_before() -> None:
    """combat_action states without a `potions` field (e.g. older fixtures)
    must never crash or spuriously emit a potion action."""
    for seed in range(50):
        rng = random.Random(seed)
        action = pick_combat_action(COMBAT_STATE, rng)
        assert action["action"] in ("play", "end_turn")


# ----------------------------------------------------------------
# run_config -> start_run policy (phase 1b)
# ----------------------------------------------------------------

RUN_CONFIG_STATE = {
    "type": "run_config",
    "request_id": "rc-1",
    "state_id": 1,
    "characters": [
        {"id": "IRONCLAD", "name": "The Ironclad"},
        {"id": "SILENT", "name": "The Silent"},
    ],
    "max_ascension": 10,
    "defaults": {"character": "IRONCLAD", "ascension": 3, "seed": None},
    "seed": "",
    "act": 1,
    "floor": 0,
    "run_state": {},
}


def test_pick_run_config_uses_defaults_when_no_cli_override() -> None:
    action = pick_run_config(RUN_CONFIG_STATE)
    assert action == {
        "action": "start_run",
        "character": "IRONCLAD",
        "ascension": 3,
        "seed": None,
    }


def test_pick_run_config_cli_overrides_win_over_defaults() -> None:
    action = pick_run_config(RUN_CONFIG_STATE, character="SILENT", ascension=7, seed=42)
    assert action == {
        "action": "start_run",
        "character": "SILENT",
        "ascension": 7,
        "seed": 42,
    }


def test_pick_run_config_falls_back_to_first_character_with_no_defaults() -> None:
    state = dict(RUN_CONFIG_STATE, defaults={})
    action = pick_run_config(state)
    assert action["character"] == "IRONCLAD"  # first entry in `characters`
    assert action["ascension"] == 0


def test_choose_action_dispatches_run_config() -> None:
    rng = random.Random(0)
    action = choose_action(
        RUN_CONFIG_STATE,
        rng,
        run_config_character="DEFECT",
        run_config_ascension=5,
        run_config_seed=99,
    )
    assert action == {
        "action": "start_run",
        "character": "DEFECT",
        "ascension": 5,
        "seed": 99,
    }


def test_choose_action_dispatches_by_type() -> None:
    rng = random.Random(0)
    assert choose_action(MAP_STATE, rng)["action"] == "choose"
    rng = random.Random(0)
    combat = choose_action(COMBAT_STATE, rng)
    assert combat["action"] in ("play", "end_turn")


def test_choose_action_unknown_type_best_effort(caplog: pytest.LogCaptureFixture) -> None:
    state = {"type": "some_future_screen", "floor": 1}
    rng = random.Random(0)
    with caplog.at_level("ERROR"):
        action = choose_action(state, rng)
    assert action == {"action": "choose", "index": 0}
    assert any("Unknown message type" in r.message for r in caplog.records)
