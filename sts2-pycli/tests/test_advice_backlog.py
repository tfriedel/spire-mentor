"""The advisor skips an advice request a newer one is queued behind (#1208).

In co-op each card a teammate plays queues another ``advice_request``; the
mod keeps only its newest one in flight and drops replies to older ones. The
serve loop answers the newest and skips the rest, so the search does not run
behind the board.
"""

from __future__ import annotations

import json
import socket
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import realgame_advisor as ra
from realgame_bridge import RealGameBridgeClient


def _serve(lines: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Queue ``lines`` from a fake mod, run the serve loop until the mod
    hangs up, and return the request ids it answered."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def mod() -> None:
        conn, _ = listener.accept()
        # One write: every line is queued before the advisor reads the first.
        conn.sendall(b"".join(json.dumps(m).encode("utf-8") + b"\n" for m in lines))
        conn.shutdown(socket.SHUT_WR)
        try:
            conn.recv(1)  # until the client disconnects
        finally:
            conn.close()

    thread = threading.Thread(target=mod, daemon=True)
    thread.start()
    answered: list[Any] = []
    monkeypatch.setattr(
        ra, "_handle_advice_request", lambda _c, *a, **k: answered.append(a[4]["request_id"])
    )
    monkeypatch.setattr(ra, "RECORDER", None)
    monkeypatch.setattr(ra, "warm_policy_async", lambda *a, **k: None)
    client = RealGameBridgeClient(host="127.0.0.1", port=port, reconnect_attempts=1, timeout=5.0)
    client.connect()
    control = SimpleNamespace(note_message=lambda m: None, set_log_path=lambda p: None)
    wlog = SimpleNamespace(path=None)
    log = SimpleNamespace(write=lambda kind, data: None)
    try:
        with pytest.raises(ConnectionError):
            ra._serve_connected(
                client, ra.RecommendConfig(), SimpleNamespace(), log, wlog, None,
                None, control, None, {}, None,
            )  # fmt: skip
    finally:
        client.disconnect()
        thread.join(timeout=5)
        listener.close()
    return answered


def _request(request_id: str, advice_for: str = "combat_action") -> dict[str, Any]:
    return {"type": "advice_request", "request_id": request_id, "advice_for": advice_for}


def test_only_the_newest_queued_request_is_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    answered = _serve([_request(str(i)) for i in range(1, 8)], monkeypatch)
    assert answered == ["7"]


def test_a_request_of_another_kind_is_still_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    # An AI partner's screen question queued behind its combat one: each side
    # waits for its own answer, so neither is skipped.
    answered = _serve([_request("1"), _request("2", "card_select")], monkeypatch)
    assert answered == ["1", "2"]
