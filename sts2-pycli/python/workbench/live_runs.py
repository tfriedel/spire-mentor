"""The one live run a client session has, and every way of replacing it.

Four routes used to each perform the same dance by hand — null the slot,
``quit()`` the prior run, construct a :class:`BranchingLiveSession`, start or
``load_line`` it, ``discard()`` on divergence, translate the exception to a
status code — and the last three restore bugs all landed inside that
duplication. :class:`LiveRuns` owns it once:

* :meth:`LiveRuns.start` / :meth:`LiveRuns.start_run_mode` — a fresh run.
* :meth:`LiveRuns.restore_save` — a named save.
* :meth:`LiveRuns.resume_recording` — "play from here" on a loaded recording.

Every one of them builds the new run **before** touching the slot, so a
rejected start or restore never costs the client the run they already had.
:class:`RunIdentity` is the one home for what a restore needs to rebuild a run
from its seed, including the legacy-save constructor probe.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workbench import saves
from workbench.branching import BranchingLiveSession
from workbench.session import Session

_log = logging.getLogger(__name__)

_Build = Callable[[], tuple[BranchingLiveSession, dict[str, Any]]]


class RestoreError(Exception):
    """A run could not be started or rebuilt. ``status`` is the HTTP status
    the server should answer with — 400 for a bad request, 404 for a missing
    step, 409 when the run itself refuses (diverged line, unresumable log)."""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class RunIdentity:
    """What rebuilds a run from its seed: the same triple a named save and a
    workbench run log carry.

    ``run_mode`` records which CONSTRUCTOR built the run — run-mode ("play to
    end") runs hash an integer seed, interactive ones use the text seed, and
    the two are different games. ``None`` means the source predates the field
    and both must be probed (see :meth:`restore_attempts`).
    """

    character: str
    seed: str
    ascension: int
    backend: str
    act1: str = saves.DEFAULT_ACT1
    run_mode: bool | None = None
    # The game build the run simulates (GH #801). Sources that predate the
    # field carry none and were all played on the default build.
    game_version: str = saves.LEGACY_GAME_VERSION

    @classmethod
    def from_save(cls, payload: dict[str, Any]) -> RunIdentity:
        """The identity a save file recorded, with the legacy defaults."""
        return cls(
            character=payload["character"],
            seed=payload["seed"],
            ascension=payload["ascension"],
            backend=payload.get("backend", saves.DEFAULT_BACKEND),
            act1=payload.get("act1", saves.DEFAULT_ACT1),
            run_mode=payload.get("run_mode", saves.UNKNOWN_RUN_MODE),
            game_version=payload.get("game_version") or saves.LEGACY_GAME_VERSION,
        )

    @classmethod
    def from_recording(cls, ident: dict[str, Any]) -> RunIdentity:
        """The identity a workbench run log's head carries
        (:meth:`~workbench.timeline.TimelineStore.run_identity`)."""
        return cls(
            character=ident["character"],
            seed=ident["seed"],
            ascension=ident["ascension"],
            backend=ident["backend"],
            act1=ident["act1"],
            run_mode=ident["run_mode"],
            game_version=ident.get("game_version") or saves.LEGACY_GAME_VERSION,
        )

    def restore_attempts(self) -> list[bool]:
        """The ``run_mode`` values to try, best first.

        One entry when the constructor is known. Legacy saves recorded none,
        so both are returned: the tell is the seed — a run-mode run stores
        ``str(int)`` while an interactive run stores the text typed — but a
        player can type digits too, so it only decides the ORDER.
        """
        if self.run_mode is not None:
            return [bool(self.run_mode)]
        likely = str(self.seed).isdigit()
        return [likely, not likely]


def resume_capability(store: Any) -> dict[str, Any]:
    """Whether (and where) "play from here" can take this recording over.

    Two independent routes, in the order :meth:`LiveRuns.resume_recording`
    tries them:

    * ``"combat"`` — a fixture-sourced combat recording (``workbench.record
      --fixture``). Its fixture can be reloaded and the logged native actions
      replayed, so any **combat** step can be taken over.
    * ``"run"`` — a workbench run log whose head carries the run identity and
      whose actions carry their replayable form. The whole run rebuilds from
      the seed, so **every** step can be taken over.

    ``kind: None`` carries a ``reason`` the UI shows instead of a dead button —
    a silent no-op is what made this feature look broken (the SPA only rendered
    ``liveError`` in play mode, so a 409 here was invisible).
    """
    fixture_path = store.fixture_path()
    # ``is_file``, not just "recorded a path": a fixture that has since moved
    # would otherwise advertise a route that dies inside the endpoint.
    if isinstance(fixture_path, str) and Path(fixture_path).is_file():
        return {"kind": "combat", "reason": None}
    # ``len(store) - 1``: the LAST state has no action after it (it is the leaf
    # the run stopped on), so asking for a line THROUGH it always comes back
    # empty-handed. The deepest resumable step is the one before it.
    if store.run_identity() is not None and store.replay_line(max(0, len(store) - 1)) is not None:
        return {"kind": "run", "reason": None}
    reason = (
        "this recording cannot be resumed: it has no readable source fixture and "
        "no replayable action line (only runs played in the workbench after "
        "2026-08-26 carry one)"
    )
    return {"kind": None, "reason": reason}


class LiveRuns:
    """Starts, restores and replaces the live run in a client's :class:`Session`.

    Bound once per app to where run logs go and the card loc map every session
    needs; each method takes the client session whose slot it fills.

    ``loc`` is the default build's map. ``loc_for`` (optional) returns the map
    for any supported build (GH #801) — a v0.111.0 run labels its log with its
    own card names; without it every build shares ``loc``.
    """

    def __init__(
        self,
        logs_dir: str | Path,
        loc: dict[str, str] | None,
        loc_for: Callable[[str], dict[str, str]] | None = None,
    ) -> None:
        self._logs_dir = str(logs_dir)
        self._loc = loc
        self._loc_for = loc_for

    def loc(self, game_version: str = saves.LEGACY_GAME_VERSION) -> dict[str, str] | None:
        """The card loc map for *game_version* (``self._loc`` is the legacy build's)."""
        if self._loc_for is None or game_version == saves.LEGACY_GAME_VERSION:
            return self._loc
        return self._loc_for(game_version)

    # -- the slot ------------------------------------------------------------
    def new_session(self, backend: str, game_version: str | None = None) -> BranchingLiveSession:
        """An unstarted session on *backend*; 400 for an unknown engine or a
        build the engine cannot run. ``game_version=None`` is the session's
        own default: the default build on the Rust engine, the process's
        oracle build on the C# one."""
        try:
            sess = BranchingLiveSession(
                logs_dir=self._logs_dir,
                backend=backend,
                game_version=game_version,
            )
        except ValueError as exc:
            raise RestoreError(str(exc), status=400) from exc
        sess._loc = self.loc(sess.game_version) or {}
        return sess

    @staticmethod
    def replace(session: Session, build: _Build) -> dict[str, Any]:
        """Install the run *build* produces as *session*'s live run.

        The build runs first, while the prior run is still whole: a start or
        restore that fails leaves the client exactly where they were. Only
        once it succeeds is the prior run quit (writing its log) and the slot
        swapped — one slot, so never two runs the client can act on.
        """
        sess, state = build()
        prior = session.live
        session.live = sess
        if prior is not None:
            try:
                prior.quit()
            except Exception:  # the new run is already installed; don't lose it
                _log.warning("quitting the prior live run failed", exc_info=True)
        return state

    # -- fresh runs -----------------------------------------------------------
    def start(
        self,
        session: Session,
        *,
        character: str,
        seed: str,
        ascension: int,
        act1: str,
        backend: str,
        game_version: str | None = None,
    ) -> dict[str, Any]:
        """A fresh interactive run, replacing any current one."""
        sess = self.new_session(backend, game_version)

        def build() -> tuple[BranchingLiveSession, dict[str, Any]]:
            try:
                return sess, sess.start(character, seed, ascension, act1)
            except (OSError, RuntimeError) as exc:
                # The engine cannot be brought up on this host. Two real shapes,
                # both of which used to 500: the .NET binary is absent entirely
                # (the deployed image ships no .NET -- OSError out of Popen), or
                # it spawns and immediately dies because the oracle was never
                # built (RuntimeError "EOF from game process"). Answer it as the
                # unavailable OPTION it is, so the SPA can render a sentence.
                sess.discard()
                raise RestoreError(
                    f"could not start the {backend!r} engine: {exc}. "
                    f"Pick another engine, or build the C# oracle (`just build-cs`) "
                    f"if you are running locally.",
                    status=400,
                ) from exc

        return self.replace(session, build)

    def start_run_mode(
        self,
        session: Session,
        *,
        character: str,
        seed: str | int,
        ascension: int,
        sims: int,
        boss_sims: int,
        threads: int,
        game_version: str | None = None,
    ) -> dict[str, Any]:
        """A run that reproduces the offline benchmark move-for-move (rust
        backend, integer seed), replacing any current one. 400 on a bad seed
        or config."""
        sess = self.new_session("rust", game_version)

        def build() -> tuple[BranchingLiveSession, dict[str, Any]]:
            try:
                state = sess.start_faithful(
                    character,
                    seed,
                    ascension=ascension,
                    sims=sims,
                    boss_sims=boss_sims,
                    threads=threads,
                )
            except (ValueError, RuntimeError) as exc:
                sess.discard()
                raise RestoreError(str(exc), status=400) from exc
            return sess, state

        return self.replace(session, build)

    # -- restores -------------------------------------------------------------
    def _rebuild(
        self,
        ident: RunIdentity,
        line: list[dict[str, Any]],
        node_values: list[float | None] | None = None,
    ) -> tuple[BranchingLiveSession, dict[str, Any]]:
        """Replay *line* into a fresh session, probing the constructor when the
        identity doesn't know it. Every failed attempt is discarded (a diverged
        restore is not a run and leaves no log); the 409 names each one."""
        failures: list[str] = []
        for run_mode in ident.restore_attempts():
            try:
                sess = self.new_session(ident.backend, ident.game_version)
            except RestoreError as exc:
                # An unknown build (a save from a newer workbench) or a build
                # the recording engine cannot run: no attempt can succeed.
                raise RestoreError(str(exc)) from exc
            try:
                state = sess.load_line(
                    ident.character,
                    ident.seed,
                    ident.ascension,
                    line,
                    run_mode=run_mode,
                    act1=ident.act1,
                    node_values=node_values,
                )
            except (RuntimeError, ValueError) as exc:
                # ValueError too: probing run_mode on a C# save hits "run-mode
                # play requires the rust backend", which is a rejected ATTEMPT,
                # not a server fault.
                sess.discard()
                failures.append(f"[run_mode={run_mode}] {exc}")
                continue
            return sess, state
        raise RestoreError("; ".join(failures))

    def restore_save(self, session: Session, payload: dict[str, Any]) -> dict[str, Any]:
        """Restore a save file's active line, replacing any current run.

        Restores on the backend that recorded the line — replaying on another
        engine diverges, since the two serialize decisions and map option
        indices differently. The recorded PPO values ride along so the net is
        not re-run at every non-combat decision (absent on older saves, which
        then recompute).
        """
        ident = RunIdentity.from_save(payload)
        return self.replace(
            session,
            lambda: self._rebuild(ident, payload["line"], payload.get("node_values")),
        )

    def resume_recording(self, session: Session, store: Any, idx: int) -> dict[str, Any]:
        """Take over a loaded recording at step *idx* ("play from here").

        Two routes, tried in the order :func:`resume_capability` reports: a
        fixture-sourced combat recording reloads its fixture and replays the
        logged native actions (combat steps only); a workbench run log rebuilds
        the whole run from its seed, so any step can be taken over. Everything
        that can refuse is settled before the current run is touched.

        Raises:
            RestoreError: 404 for an out-of-range step; 409 with the reason
            when neither route fits, or the line diverges.
        """
        try:
            step = store.get(idx)
        except IndexError as exc:
            raise RestoreError(str(exc), status=404) from exc

        cap = resume_capability(store)
        if cap["kind"] is None:
            raise RestoreError(cap["reason"])

        if cap["kind"] == "combat":
            if step.get("state", {}).get("decision") != "combat_play":
                raise RestoreError(
                    "this recording is a combat fixture, so play-from-here only "
                    "works on a combat decision"
                )
            fixture_path = store.fixture_path()
            actions = store.native_actions_before(idx)
            if not isinstance(fixture_path, str) or actions is None:
                raise RestoreError("this run cannot be replayed (no source fixture)")
            character = store.character()

            def build_combat() -> tuple[BranchingLiveSession, dict[str, Any]]:
                sess = BranchingLiveSession.from_replay(
                    logs_dir=self._logs_dir,
                    loc=self._loc,
                    fixture_path=fixture_path,
                    prefix_actions=actions,
                    character=character,
                )
                return sess, sess.start(character, "replay", 0)

            return self.replace(session, build_combat)

        ident_dict = store.run_identity()
        line = store.replay_line(idx)
        if ident_dict is None or line is None:
            raise RestoreError("this run cannot be resumed (no replayable action line)")
        ident = RunIdentity.from_recording(ident_dict)

        def build_run() -> tuple[BranchingLiveSession, dict[str, Any]]:
            try:
                return self._rebuild(ident, line)
            except RestoreError as exc:
                # Say WHERE it diverged (load_line names the step + phase).
                raise RestoreError(f"could not rebuild this run: {exc}", status=exc.status) from exc

        return self.replace(session, build_run)
