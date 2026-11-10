"""Unit tests for the M6 productionized run-parity sweep machinery.

Pure-Python — no game engine. The C# recording (`_record_cs_trajectory`) and
both backends (`make_backend`) are monkeypatched to deterministic stubs, so
these cover the sweep plumbing (divergence-signature bucketing, the trace
cache's key/namespace/hit-miss/format-invalidation, and the record→replay
equivalence path) without spawning a .NET subprocess or importing the wheel.

The live-vs-cache equivalence that actually drives both engines is exercised by
the smoke sweep in CI (`run_parity.py --sweep`), not here.
"""

from __future__ import annotations

import json
import threading

import pytest
import run_parity as rp
from run_parity import Divergence, RunResult


# --------------------------------------------------------------------------- #
# run_signature — the bucket key that groups same-root divergences across seeds
# --------------------------------------------------------------------------- #
def _diverged(fields: list[str], phase: str, kind: str) -> RunResult:
    res = RunResult(seed="S", steps=3, terminal="diverged")
    res.divergence = Divergence(3, phase, fields, kind)
    return res


def test_run_signature_clean_buckets_by_terminal() -> None:
    assert rp.run_signature(RunResult(seed="S", steps=9, terminal="max_steps")) == "clean:max_steps"
    assert (
        rp.run_signature(RunResult(seed="S", steps=9, terminal="game_over_match"))
        == "clean:game_over_match"
    )


def test_run_signature_strips_seed_specific_numbers() -> None:
    # Two seeds, same root cause, different HP values → ONE bucket.
    a = _diverged(["hp: cs=58 rs=69"], "combat", "state")
    b = _diverged(["hp: cs=24 rs=27"], "combat", "state")
    assert rp.run_signature(a) == rp.run_signature(b) == "combat/state/hp"


def test_run_signature_act_reject_keeps_verb_drops_kwargs() -> None:
    r = _diverged(
        ["act(choose_option,{'option_index': 1}) -> cs_err=None rs_err='bad'"],
        "rest",
        "act_reject",
    )
    assert rp.run_signature(r) == "rest/act_reject/act:choose_option"


def test_run_signature_phase_mismatch_keeps_labels() -> None:
    # A phase-label mismatch IS the signature — keep cs/rs labels (granularity skew).
    r = _diverged(["phase: cs='card_select' rs='map'"], "card_select", "state")
    assert rp.run_signature(r) == "card_select/state/phase:cs=card_selectrs=map"


def test_run_signature_map_node_drops_coords() -> None:
    a = _diverged(["map.node[(1, 2)]: cs=(...) rs=(...)"], "map", "map")
    b = _diverged(["map.node[(3, 4)]: cs=(...) rs=(...)"], "map", "map")
    assert rp.run_signature(a) == rp.run_signature(b) == "map/map/map.node"


# --------------------------------------------------------------------------- #
# _harness_hash + cache key
# --------------------------------------------------------------------------- #
def test_harness_hash_is_stable_8_hex() -> None:
    h = rp._harness_hash()
    assert h == rp._harness_hash()
    assert len(h) == 8
    int(h, 16)  # valid hex


def test_cache_key_identity_sensitive(tmp_path) -> None:
    c = rp._RunTraceCache(str(tmp_path), "ns")
    k = c._key("S", "Ironclad", 0, 30)
    assert k == c._key("S", "Ironclad", 0, 30)  # deterministic
    assert k != c._key("T", "Ironclad", 0, 30)  # seed
    assert k != c._key("S", "Silent", 0, 30)  # character
    assert k != c._key("S", "Ironclad", 1, 30)  # ascension
    assert k != c._key("S", "Ironclad", 0, 40)  # max_steps


# --------------------------------------------------------------------------- #
# _RunTraceCache — hit/miss, persistence, format invalidation
# --------------------------------------------------------------------------- #
def _stub_rec(seed: str = "S", character: str = "Ironclad") -> dict:
    return {
        "v": rp._RUN_TRACE_FORMAT,
        "seed": seed,
        "character": character,
        "ascension": 0,
        "max_steps": 30,
        "terminal": "max_steps",
        "entries": [],
    }


def test_cache_records_on_miss_then_serves_hit(tmp_path, monkeypatch) -> None:
    calls = {"n": 0}

    def fake_record(seed, character, ascension, max_steps):
        calls["n"] += 1
        return _stub_rec(seed, character)

    monkeypatch.setattr(rp, "_record_cs_trajectory", fake_record)
    c = rp._RunTraceCache(str(tmp_path), "nsX")

    rec1, st1 = c.get_or_record("S", "Ironclad", 0, 30)
    assert st1 == "miss" and calls["n"] == 1
    rec2, st2 = c.get_or_record("S", "Ironclad", 0, 30)
    assert st2 == "hit" and calls["n"] == 1  # not re-recorded
    assert rec1 == rec2


def test_cache_format_bump_invalidates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rp, "_record_cs_trajectory", lambda *a: _stub_rec())
    c = rp._RunTraceCache(str(tmp_path), "nsY")
    path = c._dir + "/" + c._key("S", "Ironclad", 0, 30) + ".json"
    # Write a stale-format trace by hand.
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"v": rp._RUN_TRACE_FORMAT - 1, "entries": []}, f)
    _rec, st = c.get_or_record("S", "Ironclad", 0, 30)
    assert st == "miss"  # stale format ignored → re-recorded


def test_namespace_isolates_caches(tmp_path) -> None:
    a = rp._RunTraceCache(str(tmp_path), "v1-aaaa")
    b = rp._RunTraceCache(str(tmp_path), "v2-bbbb")
    assert a._dir != b._dir  # different oracle/harness hash → different subdir


def test_replay_only_fails_closed_on_miss(tmp_path, monkeypatch) -> None:
    # The critical no-game guarantee: a miss must NOT fall through to C#.
    spawned = {"n": 0}

    def fake_record(*_a):
        spawned["n"] += 1
        return _stub_rec()

    monkeypatch.setattr(rp, "_record_cs_trajectory", fake_record)
    c = rp._RunTraceCache(str(tmp_path), "nsRO")
    with pytest.raises(rp._CacheMiss):
        c.get_or_record("S", "Ironclad", 0, 30, replay_only=True)
    assert spawned["n"] == 0  # C# recording never invoked


def test_replay_only_serves_hit_without_recording(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rp, "_record_cs_trajectory", lambda *a: _stub_rec())
    c = rp._RunTraceCache(str(tmp_path), "nsRO2")
    c.get_or_record("S", "Ironclad", 0, 30)  # warm (record)
    spawned = {"n": 0}
    monkeypatch.setattr(rp, "_record_cs_trajectory", lambda *a: spawned.update(n=1) or _stub_rec())
    _rec, st = c.get_or_record("S", "Ironclad", 0, 30, replay_only=True)
    assert st == "hit" and spawned["n"] == 0


def test_run_one_cached_replay_only_raises_when_no_cache(monkeypatch) -> None:
    monkeypatch.setattr(rp, "_RUN_CACHE", None)
    monkeypatch.setattr(rp, "_RUN_REPLAY_ONLY", True)
    with pytest.raises(rp._CacheMiss):
        rp.run_one_cached("S", "Ironclad", 0, 30)


# --------------------------------------------------------------------------- #
# record → replay equivalence (fake Rust backend, scripted views)
# --------------------------------------------------------------------------- #
class _FakeBackend:
    """Returns a scripted sequence of views for start()/act()."""

    def __init__(self, views: list[dict]):
        self._views = views
        self._i = 0

    def _next(self) -> dict:
        v = self._views[self._i]
        self._i += 1
        return v

    def start(self, **_kw) -> dict:
        return self._next()

    def act(self, _verb, **_kw) -> dict:
        return self._next()

    def get_map(self) -> dict:
        return {"rows": [], "boss": {}}

    def close(self) -> None:
        pass


def _player(hp: int) -> dict:
    return {"hp": hp, "max_hp": 80, "block": 0, "gold": 50, "deck": [], "relics": [], "potions": []}


def _event_view(hp: int) -> dict:
    return {"decision": "event_choice", "player": _player(hp)}


def _gameover_view(hp: int, victory: bool = True) -> dict:
    return {"decision": "game_over", "victory": victory, "player": _player(hp)}


_REC = {
    "v": rp._RUN_TRACE_FORMAT,
    "seed": "S",
    "character": "Ironclad",
    "ascension": 0,
    "max_steps": 30,
    "terminal": "game_over",
    "entries": [
        {
            "view": _event_view(80),
            "map": None,
            "verb": "choose_option",
            "kwargs": {"option_index": 0},
        },
        {"view": _gameover_view(75), "map": None, "verb": None, "kwargs": {}},
    ],
}


def _install_fake_rust(monkeypatch, views: list[dict]) -> None:
    # Mirror the real ``make_backend`` signature: it now accepts backend
    # kwargs (e.g. ``target_convention="csharp"`` for the rust replay backend).
    monkeypatch.setattr(rp, "make_backend", lambda which, **kwargs: lambda: _FakeBackend(views))


def test_replay_clean_when_rust_matches(monkeypatch) -> None:
    # Rust mirrors the recorded C# trajectory exactly → game_over_match, no divergence.
    _install_fake_rust(monkeypatch, [_event_view(80), _gameover_view(75)])
    res = rp._replay_against_rust(_REC)
    assert res.divergence is None
    assert res.terminal == "game_over_match"
    assert res.cs_won is True and res.rs_won is True


def test_replay_diverges_when_rust_differs(monkeypatch) -> None:
    # Rust HP drifts at the terminal view → caught as a state divergence.
    _install_fake_rust(monkeypatch, [_event_view(80), _gameover_view(70)])
    res = rp._replay_against_rust(_REC)
    assert res.divergence is not None
    assert res.terminal == "diverged"
    assert any(f.startswith("hp:") for f in res.divergence.fields)


def test_replay_detects_rust_act_rejection(monkeypatch) -> None:
    # Rust rejects the applied verb (C# did NOT) → DISAGREEMENT → act_reject divergence.
    _install_fake_rust(monkeypatch, [_event_view(80), {"type": "error", "error": "illegal"}])
    res = rp._replay_against_rust(_REC)
    assert res.divergence is not None
    assert res.divergence.kind == "act_reject"


def test_replay_act_reject_reports_rust_message_field(monkeypatch) -> None:
    # Rust signals rejection via ``{"type": "error", "message": ...}`` (no
    # ``error`` key) — the reported rs_err must surface the message, not ``None``.
    _install_fake_rust(
        monkeypatch, [_event_view(80), {"type": "error", "message": "shop_purchase_failed"}]
    )
    res = rp._replay_against_rust(_REC)
    assert res.divergence is not None and res.divergence.kind == "act_reject"
    assert "shop_purchase_failed" in res.divergence.fields[0]
    assert "rs_err=None" not in res.divergence.fields[0]


#: C# rejected the (random-policy) action — recording ended at ``cs_act_reject``.
_REC_CS_REJECT = {
    "v": rp._RUN_TRACE_FORMAT,
    "seed": "S",
    "character": "Ironclad",
    "ascension": 0,
    "max_steps": 30,
    "terminal": "cs_act_reject",
    "entries": [
        {
            "view": _event_view(80),
            "map": None,
            "verb": "buy_card",
            "kwargs": {"card_index": 5},
            "cs_err_msg": "Not enough gold",
        },
    ],
}


def test_replay_both_reject_is_clean_not_divergence(monkeypatch) -> None:
    # BOTH engines reject the same illegal move (the choice enumerator offers
    # unaffordable buys regardless of price) → AGREEMENT, not a divergence. The
    # run ends at the recorded ``cs_act_reject`` terminal and buckets as clean.
    _install_fake_rust(
        monkeypatch, [_event_view(80), {"type": "error", "message": "shop_purchase_failed"}]
    )
    res = rp._replay_against_rust(_REC_CS_REJECT)
    assert res.divergence is None
    assert res.terminal == "cs_act_reject"
    assert rp.run_signature(res) == "clean:cs_act_reject"


def test_replay_cs_reject_but_rust_accepts_is_divergence(monkeypatch) -> None:
    # C# rejected the move but Rust ACCEPTED it (e.g. a missing affordability
    # check) → DISAGREEMENT → act_reject divergence.
    _install_fake_rust(monkeypatch, [_event_view(80), _event_view(80)])
    res = rp._replay_against_rust(_REC_CS_REJECT)
    assert res.divergence is not None
    assert res.divergence.kind == "act_reject"
    assert "cs_err=Not enough gold" in res.divergence.fields[0]


# --------------------------------------------------------------------------- #
# record watchdog — a wedged C# step is abandoned, on every platform
# --------------------------------------------------------------------------- #
class _WedgingCsBackend(_FakeBackend):
    """C# stub whose act() blocks once its script runs out, the way the headless
    wedges, until close() — which is what the watchdog calls — ends it with the
    real pipe's EOF error."""

    def __init__(self, views: list[dict]):
        super().__init__(views)
        self._closed = threading.Event()

    def act(self, _verb, **_kw) -> dict:
        if self._i < len(self._views):
            return self._next()
        assert self._closed.wait(timeout=30), "watchdog never closed the backend"
        raise RuntimeError("EOF from game process")

    def close(self) -> None:
        self._closed.set()


def _event_with_option(hp: int) -> dict:
    return {**_event_view(hp), "options": [{"index": 0}]}


def test_record_timeout_abandons_a_wedged_step(monkeypatch) -> None:
    # Was SIGALRM, which Windows lacks: every record died at step 0 there.
    monkeypatch.setenv("STS2_RUNPARITY_RECORD_TIMEOUT", "1")
    backend = _WedgingCsBackend([_event_with_option(80), _event_with_option(80)])
    monkeypatch.setattr(rp, "make_backend", lambda which, **kwargs: lambda: backend)
    rec = rp._record_cs_trajectory("S", "Ironclad", 0, 30)
    assert rec["terminal"] == "record_timeout"
    assert len(rec["entries"]) == 1  # the step before the wedge is kept


def test_record_error_without_timeout_still_raises(monkeypatch) -> None:
    # Only a watchdog trip is booked as record_timeout; a real C# crash is not.
    monkeypatch.setenv("STS2_RUNPARITY_RECORD_TIMEOUT", "30")

    class _Crashing(_FakeBackend):
        def act(self, _verb, **_kw) -> dict:
            raise RuntimeError("EOF from game process")

    backend = _Crashing([_event_with_option(80)])
    monkeypatch.setattr(rp, "make_backend", lambda which, **kwargs: lambda: backend)
    with pytest.raises(RuntimeError, match="EOF"):
        rp._record_cs_trajectory("S", "Ironclad", 0, 30)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_enumerate_choices_skips_disabled_rest_option() -> None:
    view = {
        "decision": "rest_site",
        "options": [
            {"index": 0, "option_id": "HEAL", "is_enabled": True},
            {"index": 1, "option_id": "SMITH", "is_enabled": False},
        ],
    }
    assert rp.enumerate_choices(None, view) == [("choose_option", {"option_index": 0})]
