"""``workbench.live_runs`` — the one owner of a client's live run.

Tested through its interface with the shared no-subprocess fake, no server.
"""

from __future__ import annotations

import pytest
from fake_game import FakeGame
from workbench import branching, canonicalize_seed
from workbench.live_runs import LiveRuns, RestoreError, RunIdentity, resume_capability
from workbench.session import Session


@pytest.fixture(autouse=True)
def _a_legacy_oracle_process(monkeypatch):
    """The C# backend runs one build, its process's. These tests start and
    restore C# runs whose saves name no build (so: the legacy one), which only
    a process selected for the legacy build can do."""
    import sts2_client

    monkeypatch.setattr(sts2_client, "GAME_VERSION", sts2_client.LEGACY_GAME_VERSION)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(branching, "Game", FakeGame)
    logs = tmp_path / "logs"
    logs.mkdir()
    return LiveRuns(logs, loc={})


# ── RunIdentity ───────────────────────────────────────────────────────────────
def test_identity_from_save_applies_legacy_defaults():
    ident = RunIdentity.from_save(
        {"character": "Ironclad", "seed": "abc", "ascension": 3, "line": []}
    )
    assert ident.backend == "csharp"  # saves predating the field were C#-only
    assert ident.act1 == "overgrowth"
    assert ident.run_mode is None  # unknown → probe both constructors


def test_identity_restore_attempts_known_constructor_is_one_attempt():
    assert RunIdentity("I", "12", 0, "rust", run_mode=True).restore_attempts() == [True]
    assert RunIdentity("I", "12", 0, "rust", run_mode=False).restore_attempts() == [False]


def test_identity_restore_attempts_legacy_ordered_by_seed_shape():
    assert RunIdentity("I", "12345", 0, "rust").restore_attempts() == [True, False]
    assert RunIdentity("I", "words", 0, "rust").restore_attempts() == [False, True]


def test_identity_from_recording_takes_the_head_verbatim():
    ident = RunIdentity.from_recording(
        {
            "character": "Silent",
            "seed": "7",
            "ascension": 5,
            "run_mode": True,
            "act1": "underdocks",
            "backend": "rust",
        }
    )
    assert (ident.character, ident.seed, ident.ascension) == ("Silent", "7", 5)
    assert ident.run_mode is True and ident.act1 == "underdocks" and ident.backend == "rust"


# ── resume_capability ────────────────────────────────────────────────────────
class _Store:
    def __init__(self, *, fixture=None, ident=None, line=None, n=3):
        self._fixture, self._ident, self._line, self._n = fixture, ident, line, n

    def fixture_path(self):
        return self._fixture

    def run_identity(self):
        return self._ident

    def replay_line(self, idx):
        return self._line

    def __len__(self):
        return self._n


def test_resume_capability_prefers_a_present_fixture(tmp_path):
    fx = tmp_path / "f.toml"
    fx.write_text("")
    assert resume_capability(_Store(fixture=str(fx)))["kind"] == "combat"


def test_resume_capability_ignores_a_missing_fixture_file(tmp_path):
    cap = resume_capability(_Store(fixture=str(tmp_path / "gone.toml")))
    assert cap["kind"] is None and "cannot be resumed" in cap["reason"]


def test_resume_capability_run_needs_identity_and_line():
    assert resume_capability(_Store(ident={"x": 1}, line=[]))["kind"] == "run"
    assert resume_capability(_Store(ident=None, line=[]))["kind"] is None
    assert resume_capability(_Store(ident={"x": 1}, line=None))["kind"] is None


# ── the slot ──────────────────────────────────────────────────────────────────
def test_start_installs_a_run(runs):
    session = Session(sid="a")
    state = runs.start(
        session, character="Ironclad", seed="s", ascension=0, act1="overgrowth", backend="csharp"
    )
    assert state["decision"] == "map_select"
    assert session.live is not None and session.live.character == "Ironclad"


def test_start_unknown_backend_is_a_400_and_keeps_the_current_run(runs):
    session = Session(sid="a")
    runs.start(
        session, character="Ironclad", seed="s", ascension=0, act1="overgrowth", backend="csharp"
    )
    prior = session.live
    with pytest.raises(RestoreError) as exc:
        runs.start(
            session, character="Ironclad", seed="s", ascension=0, act1="overgrowth", backend="bogus"
        )
    assert exc.value.status == 400
    assert session.live is prior and not prior.game.closed


def test_replace_builds_first_then_quits_the_prior(runs):
    session = Session(sid="a")
    runs.start(
        session, character="Ironclad", seed="one", ascension=0, act1="overgrowth", backend="csharp"
    )
    prior = session.live
    prior_game = prior.game
    runs.start(
        session, character="Ironclad", seed="two", ascension=0, act1="overgrowth", backend="csharp"
    )
    assert session.live is not prior
    assert prior_game.closed and prior.game is None  # quit → engine released
    assert session.live.seed == canonicalize_seed("two")


def test_replace_failure_leaves_the_slot_untouched(runs):
    session = Session(sid="a")
    runs.start(
        session, character="Ironclad", seed="s", ascension=0, act1="overgrowth", backend="csharp"
    )
    prior = session.live

    def boom():
        raise RestoreError("nope")

    with pytest.raises(RestoreError):
        runs.replace(session, boom)
    assert session.live is prior and not prior.game.closed


# ── restores ──────────────────────────────────────────────────────────────────
def test_restore_save_replays_the_line_on_the_recorded_backend(runs):
    session = Session(sid="a")
    payload = {
        "character": "Ironclad",
        "seed": "s",
        "ascension": 0,
        "backend": "csharp",
        "run_mode": False,
        "line": [{"action": "select_map_node", "args": {"col": 0}}],
    }
    state = runs.restore_save(session, payload)
    assert state["decision"] == "combat_play"
    assert session.live.active_line()[0] == payload["line"]


def test_restore_save_probes_both_constructors_for_a_legacy_save(runs):
    """A legacy C# save (no ``run_mode``) with a digit seed tries run-mode
    first — which the C# backend refuses — then the interactive constructor."""
    session = Session(sid="a")
    payload = {"character": "Ironclad", "seed": "12345", "ascension": 0, "line": []}
    state = runs.restore_save(session, payload)
    assert state["decision"] == "map_select"
    assert session.live.run_mode is False


def test_restore_save_divergence_is_a_409_naming_every_attempt(runs):
    session = Session(sid="a")
    payload = {
        "character": "Ironclad",
        "seed": "s",
        "ascension": 0,
        "backend": "csharp",
        "run_mode": False,
        "line": [{"action": "play_card", "args": {"card_index": 99}}],  # FakeGame: illegal
    }
    with pytest.raises(RestoreError) as exc:
        runs.restore_save(session, payload)
    assert exc.value.status == 409
    assert "[run_mode=False]" in str(exc.value) and "diverged" in str(exc.value)
    assert session.live is None


def test_resume_recording_out_of_range_is_a_404(runs):
    class _S(_Store):
        def get(self, idx):
            raise IndexError(f"step {idx} out of range")

    with pytest.raises(RestoreError) as exc:
        runs.resume_recording(Session(sid="a"), _S(), 9)
    assert exc.value.status == 404


def test_resume_recording_rebuilds_a_run_log(runs):
    class _S(_Store):
        def get(self, idx):
            return {"state": {"decision": "combat_play"}}

    store = _S(
        ident={
            "character": "Ironclad",
            "seed": "s",
            "ascension": 0,
            "run_mode": False,
            "act1": "overgrowth",
            "backend": "csharp",
        },
        line=[{"action": "select_map_node", "args": {"col": 0}}],
    )
    session = Session(sid="a")
    state = runs.resume_recording(session, store, 1)
    assert state["decision"] == "combat_play"
    assert len(session.live.active_line()[0]) == 1
