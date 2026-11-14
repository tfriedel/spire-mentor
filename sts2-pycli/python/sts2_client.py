"""Python client for the STS2 headless C# process.

Wraps the JSON-over-stdio protocol with a thread-safe ``Game`` class, typed
action helpers (one per verb in ``RunSimulator.cs::ExecuteActionImpl``), and
small utilities for response post-processing. Used by ``tests/conftest.py``,
``agent/sts2_bridge.py``, and ``python/sts2_mcp.py``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterable
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.join(REPO_ROOT, "src", "Sts2Headless", "Sts2Headless.csproj")

# ──────────────────────────────────────────────────────────────────────────────
# Game version (GH #789). The C# oracle is built once per supported game build
# (`just build-cs <ver>`): DLLs in `lib/<ver>/`, binary in `bin/<ver>/…`.
# `STS2_GAME_VERSION` picks which one this process drives; it is resolved once
# at import (`GAME_VERSION`, `HEADLESS_BIN`, `LIB_DIR`) and can be moved for the
# whole process with :func:`set_game_version` (the `--game-version` CLI flags
# and the pytest option do that, before any `Game` is spawned).
# ──────────────────────────────────────────────────────────────────────────────

#: The default build: stable Steam. Moves when the beta goes stable.
DEFAULT_GAME_VERSION = "v0.107.1"
#: Every build the oracle can be compiled for (mirrors the csproj check).
SUPPORTED_GAME_VERSIONS: tuple[str, ...] = ("v0.107.1", "v0.111.0", "v0.113.0")
#: Steam beta builds whose content the engine has only partly ported: the
#: advisor answers on them, and labels the answer approximate (GH #1303).
#: Empty since v0.113.0's port landed (#1319 run level, #1330 The Fool's cards).
BETA_GAME_VERSIONS: frozenset[str] = frozenset()
GAME_VERSION_ENV = "STS2_GAME_VERSION"


def normalize_game_version(version: str) -> str:
    """``"0.111.0"`` / ``"v0.111.0"`` → ``"v0.111.0"``; rejects unsupported builds."""
    v = version.strip()
    if v and not v.startswith("v"):
        v = "v" + v
    if v not in SUPPORTED_GAME_VERSIONS:
        raise ValueError(
            f"unsupported game version {version!r} (supported: {', '.join(SUPPORTED_GAME_VERSIONS)})"
        )
    return v


def resolve_game_version() -> str:
    """The version ``STS2_GAME_VERSION`` names, or :data:`DEFAULT_GAME_VERSION`."""
    raw = os.environ.get(GAME_VERSION_ENV, "").strip()
    return normalize_game_version(raw) if raw else DEFAULT_GAME_VERSION


def _apphost_name() -> str:
    """The .NET apphost file name: ``Sts2Headless.exe`` on Windows, else ``Sts2Headless``.

    Without the suffix ``_spawn_cmd``'s ``isfile`` probe never matched on
    Windows, so every spawn there silently took the slow ``dotnet run``
    fallback even with the binary built (GH #506).
    """
    return "Sts2Headless.exe" if sys.platform == "win32" else "Sts2Headless"


def headless_bin(version: str | None = None) -> str:
    """The built apphost for ``version`` (``bin/<ver>/Debug/net9.0/Sts2Headless[.exe]``)."""
    ver = normalize_game_version(version) if version else GAME_VERSION
    return os.path.join(
        REPO_ROOT, "src", "Sts2Headless", "bin", ver, "Debug", "net9.0", _apphost_name()
    )


_LEGACY_LIB_WARNED = False


def lib_dir(version: str | None = None) -> str:
    """The game-DLL dir for ``version``: ``lib/<ver>/``.

    Compatibility fallback (pre-#789 layout): when ``lib/<ver>/`` has no
    ``sts2.dll`` but a flat ``lib/sts2.dll`` exists and the version is
    v0.107.1 — the only build that layout ever held — the flat ``lib/`` is
    returned with a :class:`DeprecationWarning`. The C# side applies the same
    fallback (``HeadlessPaths.LibDir`` / the csproj).
    """
    global _LEGACY_LIB_WARNED
    ver = normalize_game_version(version) if version else GAME_VERSION
    base = os.path.join(REPO_ROOT, "lib")
    versioned = os.path.join(base, ver)
    if (
        ver == DEFAULT_GAME_VERSION
        and not os.path.isfile(os.path.join(versioned, "sts2.dll"))
        and os.path.isfile(os.path.join(base, "sts2.dll"))
    ):
        if not _LEGACY_LIB_WARNED:
            _LEGACY_LIB_WARNED = True
            import warnings

            warnings.warn(
                f"using the legacy flat game-DLL layout {base}; move the DLLs to "
                f"lib/{ver}/ (GH #789)",
                DeprecationWarning,
                stacklevel=2,
            )
        return base
    return versioned


GAME_VERSION = resolve_game_version()
HEADLESS_BIN = headless_bin()


def set_game_version(version: str) -> str:
    """Point this process (and every child it spawns) at another game build.

    Sets ``STS2_GAME_VERSION`` and re-resolves :data:`GAME_VERSION`,
    :data:`HEADLESS_BIN` and :data:`LIB_DIR`. Call it before the first
    :class:`Game` is spawned. Returns the normalized version.
    """
    global GAME_VERSION, HEADLESS_BIN, LIB_DIR
    ver = normalize_game_version(version)
    os.environ[GAME_VERSION_ENV] = ver
    GAME_VERSION = ver
    HEADLESS_BIN = headless_bin(ver)
    LIB_DIR = lib_dir(ver)
    return ver


def add_game_version_arg(parser: Any) -> None:
    """Add the shared ``--game-version`` option to an ``argparse`` parser.

    Pair with :func:`apply_game_version_arg` right after ``parse_args``.
    """
    parser.add_argument(
        "--game-version",
        default=None,
        metavar="VERSION",
        help=(
            "game build the C# oracle (and, where the installed sts2_rs supports "
            f"it, the Rust engine) simulates: {', '.join(SUPPORTED_GAME_VERSIONS)}. "
            f"Default: ${GAME_VERSION_ENV} or {DEFAULT_GAME_VERSION}. Sets "
            f"{GAME_VERSION_ENV} for every worker and child process; needs "
            "`just build-cs <version>`."
        ),
    )


def apply_game_version_arg(args: Any) -> str:
    """Apply a parsed ``--game-version`` (if given); returns the active version."""
    requested = getattr(args, "game_version", None)
    if requested:
        try:
            return set_game_version(requested)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    return GAME_VERSION


def rs_engine_call(fn: Any, /, *args: Any, version: str | None = None, **kwargs: Any) -> Any:
    """Call an ``sts2_rs`` engine entry point for the selected game build.

    Always passes ``game_version=<ver>`` (``version`` or the selected
    :data:`GAME_VERSION`). Every ``sts2_rs`` entry point routed through here
    (``RunManager``, ``build_scenario_state``, ``roll_acts``) takes the kwarg
    since #787/#788, and passing the default ``"v0.107.1"`` explicitly is the
    same as omitting it, so a v0.107.1 call simulates exactly what it did
    before. There is deliberately no fallback for a wheel that rejects the
    kwarg (the pre-#803 shim): silently simulating v0.107.1 against a v0.111.0
    oracle turned every divergence into version skew. A stale wheel now fails
    loudly with the ``TypeError``.
    """
    ver = normalize_game_version(version) if version else GAME_VERSION
    return fn(*args, game_version=ver, **kwargs)


class GameVersionMismatch(AssertionError):
    """An engine object simulates another game build than the one asked for.

    An ``AssertionError``, not a ``ValueError``: fight builders read a
    ``ValueError`` as "this row does not build" and skip the row, so a version
    mix must not look like one.
    """


def require_game_version(version: str) -> str:
    """:func:`normalize_game_version`, raising :class:`GameVersionMismatch`.

    For a build a caller asks an engine object for: an unsupported one must
    not surface as the ``ValueError`` fight builders skip a row on.
    """
    try:
        return normalize_game_version(version)
    except ValueError as exc:
        raise GameVersionMismatch(str(exc)) from exc


def check_game_version(obj: Any, version: str | None = None, *, what: str = "engine object") -> Any:
    """Return ``obj`` once its ``game_version`` getter reads ``version``.

    ``version`` defaults to the selected :data:`GAME_VERSION`. ``RunManager``
    and ``CombatState`` both carry the getter. Rust never reads
    ``STS2_GAME_VERSION``, so an engine call without ``game_version=`` builds
    v0.107.1 whatever the selection; reading the build back is what catches it.

    Raises:
        GameVersionMismatch: ``obj`` simulates another build, has no getter,
            or ``version`` names no supported build.
    """
    ver = require_game_version(version) if version else GAME_VERSION
    got = getattr(obj, "game_version", None)
    if got != ver:
        raise GameVersionMismatch(f"{what} simulates {got!r}, expected {ver!r}")
    return obj


# A drained stderr line that means the WORLD threw, not that it logged. The C#
# side routes swallowed task exceptions through `TaskHelper.LogTaskExceptions`
# -> `GD.PrintErr`, so the throw only ever appears here — never on the wire, and
# never in the run's result. Matched on the exception type name (every .NET
# exception class ends in `Exception`) plus the two runtime-emitted headers.
# Deliberately narrow: the headless also prints benign stderr chatter (e.g.
# `LocManager ... <ValidationErrors>`), and a counter that cries wolf gets
# ignored, which is the failure mode this whole counter exists to fix.
_WORLD_ERROR_RE = re.compile(
    r"(?:^|[\s:(\[])(?:[A-Za-z0-9_.]+\.)?[A-Za-z0-9_]*Exception\b"
    r"|^\s*Unhandled exception\b"
    r"|^\s*Unobserved task exception\b"
)
# Stack-frame lines belonging to a throw already counted by its header line. They
# must be excluded BEFORE classification, because a frame's own parameter list can
# name the type: `at …NGame.ReturnToMainMenuWithInternalError(Exception e)` and
# `at …SentryService.CaptureException(Exception ex)` are both real v0.107.1 members
# and both match the pattern above. Counting them inflates a single throw into
# several — the direction that cries wolf, and the one that makes `count` not mean
# what its docstring says.
_STACK_FRAME_RE = re.compile(r"^\s*(?:\[[A-Z]+\]\s*)?at\s")


def is_world_error_line(text: str) -> bool:
    """Whether a drained stderr line should score as a world error.

    The single definition used by the drain thread — exported so the guard test
    exercises what the counter actually runs, not a re-derivation of it.
    """
    return not _STACK_FRAME_RE.match(text) and bool(_WORLD_ERROR_RE.search(text))


# Cap on retained sample lines, so a run that throws in a loop cannot grow the
# report without bound. The count is always exact; only the sample is capped.
_WORLD_ERROR_SAMPLE_MAX = 20
# Same bound, for the opt-in `watch_stderr` retention.
_STDERR_WATCH_MAX = 200

#: Seconds to wait for the C# process to answer one command before treating it
#: as hung. Some harness handlers run async game code synchronously
#: (``GetAwaiter().GetResult()``); when that code waits on work only the blocked
#: thread could run, the process sits at 0% CPU forever (round-3 Orrery shop
#: entry, round-4 Ironclad-8084-a0-act2#b1, which stalled a sweep 21 minutes).
#: ``0`` disables the deadline.
REPLY_TIMEOUT_ENV = "STS2_CS_REPLY_TIMEOUT"
DEFAULT_REPLY_TIMEOUT = 300.0


def reply_timeout() -> float:
    """The per-command reply deadline in seconds; ``0`` means none."""
    raw = os.environ.get(REPLY_TIMEOUT_ENV, "").strip()
    return float(raw) if raw else DEFAULT_REPLY_TIMEOUT


# Default search locations for game DLLs (used only when ``lib/`` does not
# already contain ``sts2.dll``; the C# binary walks up looking for it).
_GAME_DIR_CANDIDATES = (
    os.path.expanduser(
        "~/Library/Application Support/Steam/steamapps/common/"
        "Slay the Spire 2/SlayTheSpire2.app/Contents/Resources/"
        "data_sts2_macos_arm64"
    ),
    os.path.expanduser("~/.steam/steam/steamapps/common/Slay the Spire 2/data_sts2_linux"),
    os.path.expanduser("~/.local/share/Steam/steamapps/common/Slay the Spire 2/data_sts2_linux"),
)


LIB_DIR = lib_dir()


class OracleUnavailable(RuntimeError):
    """The C# oracle cannot run here: no game assembly to link against."""


class OracleHang(RuntimeError):
    """The C# process did not answer a command within :func:`reply_timeout`.

    The process has been killed. The message starts with ``EOF from game
    process`` so callers that match the dead-process message still classify it
    as one, and it names the command that hung.
    """


def oracle_available(version: str | None = None) -> bool:
    """Whether the game assembly the headless harness links against is present.

    ``lib/<ver>/sts2.dll`` (``ver`` defaults to the selected
    :data:`GAME_VERSION`) is produced by ``scripts/setup_oracle.py`` from the
    player's own game install (``just import-game``) and is never published.
    For the default build, ``STS2_GAME_DIR`` (or a default Steam location) is
    the runtime fallback the C# side uses; another build needs its own
    ``lib/<ver>/`` — a Steam install holds whichever build Steam last updated
    to, so it cannot stand in for a pinned one. A git-LFS pointer file does
    not count.
    """
    ver = normalize_game_version(version) if version else GAME_VERSION
    dll = os.path.join(lib_dir(ver), "sts2.dll")
    if os.path.isfile(dll):
        try:
            with open(dll, "rb") as fh:
                return not fh.read(24).startswith(b"version https://git-lfs")
        except OSError:
            return False
    if ver != DEFAULT_GAME_VERSION:
        return False
    game_dir = _resolve_game_dir()
    return bool(game_dir) and os.path.isfile(os.path.join(game_dir, "sts2.dll"))


def oracle_built(version: str | None = None) -> bool:
    """Whether the headless binary for ``version`` has been built (``just build-cs <ver>``).

    ``STS2_HEADLESS_BIN`` (an explicit diagnostic build) counts as built for
    the selected version. Checks the managed ``Sts2Headless.dll`` rather than
    the apphost, which is ``Sts2Headless.exe`` on Windows.
    """
    ver = normalize_game_version(version) if version else GAME_VERSION
    if ver == GAME_VERSION and os.environ.get("STS2_HEADLESS_BIN"):
        return True
    return os.path.isfile(os.path.join(os.path.dirname(headless_bin(ver)), "Sts2Headless.dll"))


def _resolve_game_dir() -> str | None:
    explicit = os.environ.get("STS2_GAME_DIR")
    if explicit:
        return explicit
    for cand in _GAME_DIR_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    return None


def _resolve_dotnet() -> str:
    arm64 = os.path.expanduser("~/.dotnet-arm64/dotnet")
    if os.path.isfile(arm64):
        return arm64
    found = shutil.which("dotnet")
    if found:
        return found
    fallback = os.path.expanduser("~/.dotnet/dotnet")
    if os.path.isfile(fallback):
        return fallback
    return "dotnet"


def _spawn_cmd() -> list[str]:
    # `STS2_HEADLESS_BIN` lets a diagnostic build (`dotnet build -o <dir>`) be
    # driven without overwriting `bin/<ver>/Debug/net9.0/`, so an instrumented
    # binary can be exercised while a long sweep is still running against the
    # normal one.
    override = os.environ.get("STS2_HEADLESS_BIN")
    if override:
        return [override]
    if os.path.isfile(HEADLESS_BIN) and os.access(HEADLESS_BIN, os.X_OK):
        return [HEADLESS_BIN]
    return [
        _resolve_dotnet(),
        "run",
        "--no-build",
        "--project",
        PROJECT,
        f"--property:GameVersion={GAME_VERSION}",
    ]


def headless_cmd() -> list[str]:
    """The argv that spawns the selected game build's headless oracle.

    ``STS2_HEADLESS_BIN`` if set, else the built ``bin/<ver>/`` apphost, else
    ``dotnet run --no-build`` with ``--property:GameVersion=<ver>``.
    """
    return _spawn_cmd()


def headless_available() -> bool:
    """Whether the C# headless engine can actually be spawned on this host.

    ``Game()`` spawns eagerly, so an absent engine surfaces as a
    ``FileNotFoundError`` out of ``Popen`` — which is what the deployed
    workbench hit: its image is deliberately C#-free (root ``Dockerfile``
    stage 3 — no .NET runtime, no game DLLs, no ``src/Sts2Headless``), yet the
    engine dropdown still offered "C#". Callers use this to decide whether to
    OFFER the backend at all, rather than letting the player pick a dead one.

    Checks only what :func:`_spawn_cmd` would run: the pinned/built binary must
    exist and be executable, or ``dotnet`` must resolve AND the csproj be
    present for the ``dotnet run`` fallback. It does NOT probe the game DLLs —
    a missing ``sts2.dll`` is a different (and much louder) failure.
    """
    cmd = _spawn_cmd()
    if len(cmd) == 1:
        return os.path.isfile(cmd[0]) and os.access(cmd[0], os.X_OK)
    dotnet = cmd[0]
    resolved = os.path.isfile(dotnet) or shutil.which(dotnet) is not None
    return resolved and os.path.isfile(PROJECT)


# ──────────────────────────────────────────────────────────────────────────────
# Response post-processing utilities.
# ──────────────────────────────────────────────────────────────────────────────

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_json(obj: Any) -> Any:
    """Recursively strip ASCII control characters from string values."""
    if isinstance(obj, str):
        return _CONTROL_CHARS.sub("", obj)
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return obj


_COMPACT_STRIP_KEYS = frozenset(
    {
        "description",
        "after_upgrade",
        "enchantment",
        "enchantment_amount",
        "affliction",
        "affliction_amount",
        "id",
        "draw_pile_count",
        "discard_pile_count",
        "upgraded",
        "act_name",
    }
)


def compact_json(obj: Any, _depth: int = 0) -> Any:
    """Trim verbose fields from a state dict for context-constrained agents.

    Mirrors the filter the HTTP bridge applies when ``--compact`` is set.
    """
    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for k, v in obj.items():
            if k in _COMPACT_STRIP_KEYS:
                continue
            if k == "context" and obj.get("decision") == "combat_play":
                continue
            if k == "player" and obj.get("decision") == "combat_play":
                if isinstance(v, dict):
                    potions = v.get("potions")
                    if potions:
                        result["potions"] = compact_json(potions, _depth + 1)
                continue
            if k == "relics" and isinstance(v, list) and _depth > 0:
                result[k] = [compact_json(r, _depth + 1) if isinstance(r, dict) else r for r in v]
                continue
            result[k] = compact_json(v, _depth + 1)
        return result
    if isinstance(obj, list):
        return [compact_json(v, _depth + 1) for v in obj]
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Action vocabulary — keep in sync with ``RunSimulator.cs::ExecuteActionImpl``.
# ──────────────────────────────────────────────────────────────────────────────

ACTIONS: tuple[str, ...] = (
    "select_map_node",
    "play_card",
    "end_turn",
    "choose_option",
    "select_card_reward",
    "skip_card_reward",
    "buy_card",
    "buy_relic",
    "buy_potion",
    "remove_card",
    "select_bundle",
    "select_cards",
    "skip_select",
    "use_potion",
    "discard_potion",
    "leave_room",
    "proceed",
)


# ──────────────────────────────────────────────────────────────────────────────
# Game process wrapper.
# ──────────────────────────────────────────────────────────────────────────────


class Game:
    """Wraps the headless C# process. Thread-safe; one game per instance."""

    def __init__(self, *, forward_stderr: bool = False, watch_stderr: str | None = None):
        """
        Args:
            forward_stderr: Echo each drained child stderr line to our own.
            watch_stderr: Regex; drained lines matching it are RETAINED and
                readable via :meth:`stderr_watch`. ``_WORLD_ERROR_RE`` only
                catches lines naming an exception TYPE, so a world failure that
                does not throw is invisible to it -- e.g.
                ``[SIM] Event SlipperyBridge didn't advance, force-finishing``,
                where the option task parked rather than faulted and the state
                effect was silently dropped. The event/option coverage harness
                (GH #313) must assert on those, so it needs a way to see any
                marker, not just exceptions.
        """
        if not oracle_available():
            raise OracleUnavailable(
                f"no game assembly for {GAME_VERSION}: lib/{GAME_VERSION}/sts2.dll is missing"
                + (" and STS2_GAME_DIR is unset" if GAME_VERSION == DEFAULT_GAME_VERSION else "")
                + " — run `just import-game` (see docs/bring-your-own-game.md)"
            )
        env = os.environ.copy()
        env[GAME_VERSION_ENV] = GAME_VERSION
        # The Steam-install fallback only for the default build: an auto-detected
        # install holds whatever build Steam last updated to.
        game_dir = _resolve_game_dir() if GAME_VERSION == DEFAULT_GAME_VERSION else None
        if game_dir:
            env["STS2_GAME_DIR"] = game_dir
        self.proc = subprocess.Popen(
            _spawn_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=REPO_ROOT,
        )
        self._lock = threading.Lock()
        self.step = 0
        self._reply_timeout = reply_timeout()
        #: Set by the watchdog when it kills a hung process: what was pending.
        self._hung: str | None = None

        # Tally of drained stderr lines that indicate the WORLD threw. Written
        # by the drain thread, read by whoever owns the Game — see
        # `world_errors()`. Guarded by its own lock so reading mid-run can never
        # contend with `send()`.
        self._err_lock = threading.Lock()
        self._world_error_count = 0
        self._world_error_sample: list[str] = []
        self._watch_re = re.compile(watch_stderr) if watch_stderr else None
        self._watch_hits: list[str] = []

        # ALWAYS drain the child's stderr on a daemon thread — never leave the
        # PIPE unread. The headless can emit unbounded error logs (e.g. a task
        # exception swallowed by TaskHelper.LogTaskExceptions, routed through
        # Godot.GD.PrintErr). If nobody reads stderr, the OS pipe buffer (~64 KB)
        # fills, the child BLOCKS forever inside Console.Write, and the game
        # thread — waiting on that task — deadlocks at 0% CPU. (This is what hung
        # the parity sweep on the TUNNELER/INKLET StratagemPower.AfterShuffle
        # trials.) Draining keeps it flowing; `forward_stderr` only controls
        # whether we echo each line to our own stderr.
        #
        # Draining is NOT the same as observing: for as long as the drain simply
        # discarded, a C# throw mid-combat was invisible to every sweep — the run
        # just wedged and got scored as a loss (the 2026-07-27 Kaiser Crab arm
        # deaths). Classify while draining so the count is always available,
        # whether or not anybody asked for the echo.
        def _drain() -> None:
            assert self.proc.stderr is not None
            for line in self.proc.stderr:
                text = line.rstrip()
                if is_world_error_line(text):
                    with self._err_lock:
                        self._world_error_count += 1
                        if len(self._world_error_sample) < _WORLD_ERROR_SAMPLE_MAX:
                            self._world_error_sample.append(text)
                if self._watch_re is not None and self._watch_re.search(text):
                    with self._err_lock:
                        if len(self._watch_hits) < _STDERR_WATCH_MAX:
                            self._watch_hits.append(text)
                if forward_stderr:
                    print(f"[sts2] {text}", file=sys.stderr)

        self._drain_thread = threading.Thread(target=_drain, daemon=True)
        self._drain_thread.start()

        ready = self._read_deadline("startup")
        if ready.get("type") != "ready":
            raise RuntimeError(f"Expected ready, got: {ready}")
        # Builds before GH #789 do not report a version; they are all v0.107.1.
        built_for = ready.get("game_version", DEFAULT_GAME_VERSION)
        if built_for != GAME_VERSION:
            self.close()
            raise RuntimeError(
                f"the headless oracle was built for {built_for} but {GAME_VERSION} was "
                f"selected ({GAME_VERSION_ENV}); build it with `just build-cs {GAME_VERSION}`"
            )
        #: The game build the running oracle links against.
        self.game_version: str = built_for

    # ── Observability ────────────────────────────────────────────────────────

    def world_errors(self, *, drain_timeout: float = 1.0) -> dict[str, Any]:
        """Exception lines the C# world printed to stderr during this session.

        A non-zero count means the game process threw. It does NOT necessarily
        mean the run is invalid — `TaskHelper.LogTaskExceptions` swallows the
        throw, so the run may carry on — but any result from such a run is
        suspect and must not be pooled with clean ones without saying so.

        ``count`` is matching LINES, not distinct throws: one NRE typically
        prints 2-3 of them (a `[WARN] Unobserved task exception` header, an
        `[ERROR] System.…Exception` line, sometimes a `[SIM]` echo). Read it as
        a presence/volume signal; use ``sample`` (or --forward-stderr) to see
        what actually threw. Stack-FRAME lines are excluded (see
        :data:`_STACK_FRAME_RE`), so a trace whose frames happen to name an
        `Exception`-typed parameter no longer inflates its own throw.

        It is also a floor on "runs that are not clean measurements", not a
        measure of it: the counter is keyed on exception TYPE NAMES, so the
        world's non-throwing failures — the harness's own
        `[SIM] Event <X> didn't advance, force-finishing`, half-applied event
        options, no-op'd effects — score as clean here.

        Call this BEFORE killing the process where possible; when the process is
        already gone, the drain thread is joined first so lines still in flight
        are counted rather than raced away.

        Args:
            drain_timeout: Seconds to wait for the drain thread to finish, when
                the child has already exited. Ignored while it is still running.

        Returns:
            ``{"count": int, "sample": [str, ...]}`` — an exact count and up to
            :data:`_WORLD_ERROR_SAMPLE_MAX` verbatim lines.
        """
        if self.proc.poll() is not None:
            self._drain_thread.join(timeout=drain_timeout)
        with self._err_lock:
            return {"count": self._world_error_count, "sample": list(self._world_error_sample)}

    def stderr_watch(self, *, clear: bool = False) -> list[str]:
        """Retained stderr lines matching this Game's ``watch_stderr`` regex.

        Empty when ``watch_stderr`` was not supplied. Capped at
        :data:`_STDERR_WATCH_MAX` lines so a looping fault cannot grow without
        bound.

        Args:
            clear: Drop the retained lines after reading, so a caller stepping
                through a run can attribute hits to the step that produced them
                rather than to the whole session.
        """
        with self._err_lock:
            hits = list(self._watch_hits)
            if clear:
                self._watch_hits.clear()
        return hits

    # ── Low-level I/O ────────────────────────────────────────────────────────

    def _read(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline().strip()
            if not line:
                if self._hung is not None:
                    raise OracleHang(
                        f"EOF from game process: no reply to {self._hung} within "
                        f"{self._reply_timeout:g}s (C# deadlock?); process killed "
                        f"({REPLY_TIMEOUT_ENV} sets the limit)"
                    )
                raise RuntimeError("EOF from game process")
            if line.startswith("{"):
                return json.loads(line)

    def _read_deadline(self, pending: str) -> dict[str, Any]:
        """:meth:`_read` under the reply watchdog.

        A timer kills the process if no reply arrives in time; the kill EOFs
        stdout, so the blocked ``readline`` returns and :meth:`_read` raises
        :class:`OracleHang`. Killing is the only portable way to unblock a
        buffered text-pipe ``readline``.
        """
        if self._reply_timeout <= 0:
            return self._read()

        def _fire() -> None:
            self._hung = pending
            self.proc.kill()

        timer = threading.Timer(self._reply_timeout, _fire)
        timer.daemon = True
        timer.start()
        try:
            return self._read()
        finally:
            timer.cancel()

    def send(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """Send a raw command and return the response.

        Raises:
            OracleHang: the process gave no reply within :func:`reply_timeout`.
        """
        with self._lock:
            assert self.proc.stdin is not None
            self.step += 1
            self.proc.stdin.write(json.dumps(cmd) + "\n")
            self.proc.stdin.flush()
            what = cmd.get("cmd", "?")
            if cmd.get("action") is not None:
                what = f"{what} {cmd['action']}"
            return self._read_deadline(f"step {self.step} ({what})")

    # ── High-level commands ──────────────────────────────────────────────────

    def start(
        self,
        character: str = "Ironclad",
        seed: str = "test",
        ascension: int = 0,
        lang: str = "en",
        played_save: bool = False,
        act1: str | None = None,
        interactive_potion_swap: bool = False,
    ) -> dict[str, Any]:
        """Start a new headless run.

        Args:
            played_save: Opt-in. When True, mimics a "played" save (many
                completed runs, NEOW_EPOCH earned but never manually
                revealed on the Timeline screen) instead of a brand-new
                profile: the run opens on ``map_select`` like a real
                lived-in save instead of the Neow blessing altar. Default
                False preserves the original fresh-save behavior
                byte-for-byte (existing goldens/tests are unaffected).
                See ``RunSimulator.BuildUnlockState`` (C#) for the full
                mechanism.
            act1: Which Act-1 variant to run -- ``"overgrowth"``,
                ``"underdocks"``, or ``"roll"`` to reproduce the real lobby's
                seed-driven pick off the ``act_selection`` stream. STS2 ships
                TWO acts at index 0 and the roll happens in ``StartRunLobby``,
                which this harness bypasses. None means ``"roll"``, as in the
                real game with everything unlocked and discovered (it used to
                mean Overgrowth on every seed).
            interactive_potion_swap: GH #1069. A potion reward that finds a
                full belt parks as a ``potion_swap`` decision (discard a held
                potion to take it, or decline), as the real reward screen
                allows. Default False keeps the historical silent drop.
        """
        cmd: dict[str, Any] = {
            "cmd": "start_run",
            "character": character,
            "seed": seed,
            "ascension": ascension,
            "lang": lang,
        }
        if played_save:
            cmd["played_save"] = True
        if act1 is not None:
            cmd["act1"] = act1
        if interactive_potion_swap:
            cmd["interactive_potion_swap"] = True
        return self.send(cmd)

    def act(self, action: str, **args: Any) -> dict[str, Any]:
        cmd: dict[str, Any] = {"cmd": "action", "action": action}
        # Drop None values so optional helper args don't get sent literally.
        non_null = {k: v for k, v in args.items() if v is not None}
        if non_null:
            cmd["args"] = non_null
        return self.send(cmd)

    def get_map(self) -> dict[str, Any]:
        return self.send({"cmd": "get_map"})

    def dump_content(self) -> dict[str, Any]:
        """Enumerate every relic / potion / card with description + resolve vars.

        Used by the placeholder-regression test to verify no offered text leaks
        an unresolved ``{Var}`` template token. Requires a started run so the
        localization tables and ModelDb content are loaded.
        """
        return self.send({"cmd": "dump_content"})

    def dump_events(self) -> dict[str, Any]:
        """Enumerate every event in ``ModelDb.AllEvents``.

        The authoritative roster for the event/option coverage harness (GH
        #313), which must not wait for a play policy to roll an event before it
        can exercise it. Requires a started run.
        """
        return self.send({"cmd": "dump_events"})

    def load_save(self, *, path: str | None = None, json_str: str | None = None) -> dict[str, Any]:
        cmd: dict[str, Any] = {"cmd": "load_save"}
        if path is not None:
            cmd["path"] = path
        if json_str is not None:
            cmd["json"] = json_str
        return self.send(cmd)

    # Test/debug-only helpers.
    def set_player(self, **kwargs: Any) -> dict[str, Any]:
        return self.send({"cmd": "set_player", **kwargs})

    def enter_room(self, room_type: str, **kwargs: Any) -> dict[str, Any]:
        return self.send({"cmd": "enter_room", "type": room_type, **kwargs})

    def set_draw_order(self, cards: Iterable[str]) -> dict[str, Any]:
        return self.send({"cmd": "set_draw_order", "cards": list(cards)})

    def legal_actions(self) -> dict[str, Any]:
        """Enumerate legal (truly-playable) actions for the current in-combat state.

        Returns ``{"actions": [{"action": "play_card", "card_index": int,
        "target_index": int | None}, ..., {"action": "end_turn"}]}``.

        The C# side applies the full ``CanPlay`` predicate, so every emitted
        entry is actually playable right now. ``target_index`` is into the
        alive-enemy subset of ``pcs.Enemies`` (matches the index ``play_card``
        expects).
        """
        return self.send({"cmd": "legal_actions"})

    def get_state(self) -> dict[str, Any]:
        """Return a normalized snapshot of the current combat state.

        See ``parity_fuzz_state.normalize_csharp`` (Task 5) for the schema.
        """
        return self.send({"cmd": "get_state"})

    def write_continue_save(self, path: str) -> dict[str, Any]:
        return self.send({"cmd": "write_continue_save", "path": path})

    # ── Typed action helpers (one per verb in ACTIONS) ───────────────────────

    def select_map_node(self, col: int, row: int) -> dict[str, Any]:
        return self.act("select_map_node", col=col, row=row)

    def play_card(self, card_index: int, target_index: int | None = None) -> dict[str, Any]:
        return self.act("play_card", card_index=card_index, target_index=target_index)

    def end_turn(self) -> dict[str, Any]:
        return self.act("end_turn")

    def choose_option(self, option_index: int) -> dict[str, Any]:
        return self.act("choose_option", option_index=option_index)

    def select_card_reward(self, card_index: int) -> dict[str, Any]:
        return self.act("select_card_reward", card_index=card_index)

    def skip_card_reward(self) -> dict[str, Any]:
        return self.act("skip_card_reward")

    def buy_card(self, card_index: int) -> dict[str, Any]:
        return self.act("buy_card", card_index=card_index)

    def buy_relic(self, relic_index: int) -> dict[str, Any]:
        return self.act("buy_relic", relic_index=relic_index)

    def buy_potion(self, potion_index: int) -> dict[str, Any]:
        return self.act("buy_potion", potion_index=potion_index)

    def remove_card(self) -> dict[str, Any]:
        return self.act("remove_card")

    def select_bundle(self, bundle_index: int) -> dict[str, Any]:
        return self.act("select_bundle", bundle_index=bundle_index)

    def select_cards(self, indices: str) -> dict[str, Any]:
        return self.act("select_cards", indices=indices)

    def skip_select(self) -> dict[str, Any]:
        return self.act("skip_select")

    def use_potion(self, potion_index: int, target_index: int | None = None) -> dict[str, Any]:
        return self.act("use_potion", potion_index=potion_index, target_index=target_index)

    def discard_potion(self, potion_index: int) -> dict[str, Any]:
        return self.act("discard_potion", potion_index=potion_index)

    def leave_room(self) -> dict[str, Any]:
        return self.act("leave_room")

    def proceed(self) -> dict[str, Any]:
        return self.act("proceed")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        try:
            if self.proc.stdin is not None and not self.proc.stdin.closed:
                self.proc.stdin.write('{"cmd":"quit"}\n')
                self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    # ── Auto-play helpers ────────────────────────────────────────────────────

    def auto_combat(self, state: dict[str, Any]) -> dict[str, Any]:
        """Play one card or end turn."""
        hand = state.get("hand", [])
        energy = state.get("energy", 0)
        playable = [c for c in hand if c.get("can_play") and c.get("cost", 99) <= energy]
        if playable:
            card = playable[0]
            target = None
            if card.get("target_type") == "AnyEnemy":
                enemies = state.get("enemies", [])
                if enemies:
                    target = enemies[0]["index"]
            return self.play_card(card["index"], target)
        return self.end_turn()

    def auto_play_combat(self, state: dict[str, Any], max_steps: int = 300) -> dict[str, Any]:
        """Auto-play combat until it ends."""
        for _ in range(max_steps):
            if state.get("decision") != "combat_play":
                return state
            state = self.auto_combat(state)
        raise RuntimeError("Combat did not end within max_steps")

    def skip_neow(self, state: dict[str, Any]) -> dict[str, Any]:
        """Skip the Neow event and all follow-up rewards until map_select."""
        for _ in range(20):
            dec = state.get("decision", "")
            if dec == "map_select":
                return state
            if dec == "event_choice":
                opts = [o for o in state["options"] if not o.get("is_locked")]
                state = self.choose_option(opts[0]["index"])
            elif dec == "card_reward":
                state = self.skip_card_reward()
            elif dec == "bundle_select":
                state = self.select_bundle(0)
            elif dec == "card_select":
                if state.get("min_select", 0) == 0:
                    state = self.skip_select()
                else:
                    state = self.select_cards("0")
            else:
                state = self.proceed()
        return state
