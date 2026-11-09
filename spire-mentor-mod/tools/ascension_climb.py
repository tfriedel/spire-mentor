#!/usr/bin/env python
"""Drive the F7 advisor lane through the ascension ladder, unattended.

One live run per game launch. Winning a run *at your current max ascension*
is what raises it (`ProgressSaveManager.IncrementSingleplayerAscension`,
v0.107.1: `if (run.Ascension == charStats.MaxAscension) MaxAscension++`, capped
at 10), so the loop is: read the profile, configure the mod for that exact
level, play, read the profile again.

The profile -- not the log -- is the authority on whether a run counted. A log
can say `has won a run with ancient` while the save says otherwise (a run
started at the wrong ascension does not increment), and a run the watchdog
killed can still have been a win. So every outcome is graded by re-reading
`progress.save`.

Usage:
    python spire-mentor-mod/tools/ascension_climb.py            # climb until all 10s
    python spire-mentor-mod/tools/ascension_climb.py --status   # print the ladder
    python spire-mentor-mod/tools/ascension_climb.py --once     # a single run
    python spire-mentor-mod/tools/ascension_climb.py --character Defect
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GAME_DIR = Path(
    os.environ.get(
        "STS2_GAME_DIR",
        r"C:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2",
    )
)
MOD_DIR = GAME_DIR / "mods" / "SpireMentor"
CONF = MOD_DIR / "SpireMentor.conf"
APPDATA = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
GODOT_LOG = APPDATA / "SlayTheSpire2" / "logs" / "godot.log"


def _is_mod_line(line: str) -> bool:
    return "[SpireMentor]" in line


def all_progress_saves() -> list[Path]:
    """Every modded profile's `progress.save` (any Steam account)."""
    root = APPDATA / "SlayTheSpire2" / "steam"
    return [p for p in root.glob("*/modded/profile*/saves/progress.save") if p.is_file()]


#: The game names the profile it plays every time it saves it:
#: `Wrote N bytes to path=user://steam/<id>/modded/profile1/saves/progress.save`.
PROFILE_WRITE_RE = re.compile(
    r"path=user://steam/(\d+)/modded/(profile\d+)/saves/progress\.save"
)


def profile_in_log(text: str) -> Path | None:
    """The `progress.save` the game in `text` actually wrote, newest write wins.

    `detect_progress_save` can only guess before launch (newest mtime), and the
    guess goes stale whenever the game loads a different slot: 2026-10-19 it
    picked profile2 while the game played profile1, so a won run came back
    UNGRADED. The log is authoritative once the game has saved once.
    """
    found = PROFILE_WRITE_RE.findall(text)
    if not found:
        return None
    steam_id, profile = found[-1]
    root = APPDATA / "SlayTheSpire2" / "steam"
    return root / steam_id / "modded" / profile / "saves" / "progress.save"


def detect_progress_save() -> Path:
    """The profile the GAME is actually playing, not the one we hard-coded.

    Which save slot the game loads is a game-side setting, and it moves: the
    modded tree here holds `profile1` (the ladder climb) and `profile2` (the
    advisor lane), and on 2026-08-12 the game was on profile2 while this driver
    still read profile1. Nothing failed loudly -- `ladder()` reported another
    profile's ascensions and every run came back UNGRADED, because the run
    history record the game had just written was in a directory nobody looked
    at. So detect it: newest `progress.save` under the MODDED tree wins (the
    mod only runs there, and the loaded profile is rewritten every room, so the
    newest is the live one). `STS2_PROGRESS_SAVE` still overrides.
    """
    env = os.environ.get("STS2_PROGRESS_SAVE")
    if env:
        return Path(env)
    root = APPDATA / "SlayTheSpire2" / "steam"
    saves = all_progress_saves()
    if saves:
        return max(saves, key=lambda p: p.stat().st_mtime)
    # No profile found: keep the pattern shape (SteamID64 segment varies per
    # account) so downstream read errors name the location that was searched.
    return root / "<steam-id>/modded/profile1/saves/progress.save"


PROGRESS = detect_progress_save()
OUT_DIR = REPO / "logs" / "ascension_climb"
LEDGER = OUT_DIR / "climb.jsonl"
MAX_ASCENSION = 10

STEAM_EXE = Path(
    os.environ.get("STS2_STEAM_EXE", r"C:/Program Files (x86)/Steam/steam.exe")
)
#: Slay the Spire 2 on Steam (steamapps/appmanifest_2868840.acf).
STEAM_APPID = os.environ.get("STS2_STEAM_APPID", "2868840")
#: How long to wait for a cold Steam to finish logging in. It took ~6 s on the
#: box this was written on; the margin is for a client that decides to update.
STEAM_READY_TIMEOUT_S = 240
#: Grace before "the game process is gone" counts as a crash. `-applaunch`
#: hands the request to Steam and returns immediately, so for the first few
#: seconds of an attempt there is legitimately no SlayTheSpire2.exe to find.
#: Without this every attempt would be declared a crash on its first poll.
GAME_APPEAR_GRACE_S = 90

# Order matters only for readability of the ledger: Ironclad is closest to the
# cap and is the character the search is best at, so finishing it first turns
# the earliest runs into visible ladder movement.
CLIMB_ORDER = ["Ironclad", "Defect", "Silent", "Regent", "Necrobinder"]
CHARACTER_IDS = {
    "Ironclad": "CHARACTER.IRONCLAD",
    "Silent": "CHARACTER.SILENT",
    "Regent": "CHARACTER.REGENT",
    "Necrobinder": "CHARACTER.NECROBINDER",
    "Defect": "CHARACTER.DEFECT",
}

# A boss think is 4000 sims -- tens of seconds with animations. Three minutes
# without a single committed action is an order of magnitude above that, and is
# the signature of a wedge (a screen nothing describes), not of slow search.
STALL_SECONDS = 180
# Reaching the first decision needs a menu walk plus the advisor's first net
# load. Well clear of the ~40 s observed.
STARTUP_SECONDS = 300
# A full winning run took ~45 min at 1000/4000 sims. Three hours is a backstop
# against a run that neither wins, dies, nor stops committing actions.
RUN_CAP_SECONDS = 3 * 60 * 60
# Total wall-clock a single run may consume across all its resume attempts.
RUN_BUDGET_SECONDS = 5 * 60 * 60
POLL_SECONDS = 20
# A wedged renderer writes `Parameter "particles" is null` at ~30 MB/s while
# the run does not advance a single room -- 1.3 GB of godot.log in the four
# minutes it took to notice by hand (floor 13, 2026-08-06). Waiting out
# STALL_SECONDS on that costs gigabytes, so growth this large with no
# committed action is its own, much faster, stall signal.
FLOOD_BYTES = 40 * 1024 * 1024
# How many times to come back for the same run after a wedge. The game writes
# current_run.save every room, so a resume costs one room, not the run --
# but a run that wedges on the same room every time must not loop forever.
RESUME_ATTEMPTS = 3

TAKEOVER = "Takeover executed"
VICTORY_RE = re.compile(r"has won a run with ancient")
GAMEOVER_RE = re.compile(r"Decision up: NGameOverScreen|has lost to encounter")
#: Where THIS launch's run begins: a new run embarks, a resumed one is
#: continued (SimAutoSlayer.TryResumeSavedRunAsync). Verdicts only count after
#: it -- see `this_run`.
RUN_START_RE = re.compile(
    r"Embarking on a singleplayer|\[SimAutoSlayer\] resume_run: continuing the saved run"
)


def this_run(text: str) -> str:
    """The part of the log after this launch's run started; "" before that.

    Autopilot abandons any saved run at the main menu, and the game books an
    abandoned run as a defeat -- `CHARACTER.SILENT has lost to encounter
    ENCOUNTER.CORPSE_SLUGS_WEAK` -- BEFORE the new run embarks. Matched against
    the whole log, that line scored a healthy Ironclad run as a loss on floor
    12 and killed it (2026-10-19, logs/2026-10-19-v0111-live run 001).
    """
    starts = list(RUN_START_RE.finditer(text))
    return text[starts[-1].start():] if starts else ""


RUNAWAY_RE = re.compile(r"has not moved after")
REFUSED_RE = re.compile(r"changed nothing on the board|the game refuses")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def say(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "climb.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------


def ladder(progress: Path | None = None) -> dict[str, int]:
    """max_ascension per character name, from the live profile save.

    Retried, because the game rewrites this file on every room transition and
    reading it mid-write yields half a JSON document. An unhandled
    JSONDecodeError here would end a multi-day climb over a torn read.
    """
    progress = progress or PROGRESS
    last: Exception | None = None
    for _ in range(10):
        try:
            data = json.loads(progress.read_text(encoding="utf-8"))
            by_id = {c["id"]: c for c in data.get("character_stats", [])}
            return {
                name: int(by_id.get(cid, {}).get("max_ascension", 0))
                for name, cid in CHARACTER_IDS.items()
            }
        except (json.JSONDecodeError, OSError) as exc:
            last = exc
            time.sleep(2)
    raise RuntimeError(f"could not read {progress}: {last}")


def wins_remaining(state: dict[str, int]) -> int:
    return sum(max(0, MAX_ASCENSION - v) for v in state.values())


def next_target(state: dict[str, int], only: str | None) -> str | None:
    if only:
        return only if state.get(only, MAX_ASCENSION) < MAX_ASCENSION else None
    for name in CLIMB_ORDER:
        if state.get(name, MAX_ASCENSION) < MAX_ASCENSION:
            return name
    return None


# --------------------------------------------------------------------------
# game process
# --------------------------------------------------------------------------


def powershell(script: str) -> str:
    """Run a PowerShell one-liner and return its stdout.

    `stdin=DEVNULL` and a timeout are not belt-and-braces: launched from a
    detached/`nohup`ed parent the child inherits a stdin that never yields, and
    a probe that blocks there hangs the whole climb silently -- observed
    2026-08-06, the driver sat two minutes past a dead game without noticing.
    A watchdog that can itself wedge is not a watchdog.
    """
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=90,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        say(f"  (powershell probe failed: {exc})")
        return ""
    return (proc.stdout or "").strip()


def game_alive() -> bool:
    out = powershell(
        "(Get-Process SlayTheSpire2 -ErrorAction SilentlyContinue | Measure-Object).Count"
    )
    try:
        return int(out) > 0
    except ValueError:
        # An unreadable probe means "don't know", not "dead" -- calling a
        # healthy run crashed would kill it. A genuinely dead game stops
        # committing actions, and the stall watch catches that anyway.
        return True


def kill_everything() -> None:
    """The game, the advisor it spawned, and the build children uv leaves behind.

    An advisor that outlives its game reconnects to the NEXT launch's port 9002
    and makes the mod think a client was there from boot -- the trap that cost a
    whole attempt on 2026-08-06. Killing it is not optional cleanup.

    The `uv\\cache\\builds` clause is the other half of that lesson. Killing an
    advisor that is mid-`uv run` orphans the build subprocesses it spawned; they
    run as `%LOCALAPPDATA%\\uv\\cache\\builds-v0\\.tmpXXXX\\Scripts\\python.exe`,
    so a filter on `*realgame_advisor*` -- which is all this function used to
    have -- misses every one of them. They then hold uv's distribution-cache
    lock, and because uv's lock timeout (300 s) equals this driver's startup
    window, both sides give up simultaneously and nothing ever logs "waiting for
    a lock". Four consecutive runs died that way on 2026-08-06 (issue 8).
    """
    powershell(
        "Stop-Process -Name SlayTheSpire2 -Force -ErrorAction SilentlyContinue; "
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='uv.exe'\" | "
        "Where-Object { $_.CommandLine -like '*realgame_advisor*' -or "
        "$_.CommandLine -like '*uv\\cache\\builds*' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    time.sleep(3)


def ensure_advisor_wheel() -> bool:
    """Build the Rust extension once, here, before any run is at stake.

    The mod launches the advisor as `uv run python realgame_advisor.py`, and uv
    rebuilds the editable `sts2-rs` extension whenever a branch switch, pull or
    stash rewrites mtimes under it. That build outlasts the 300 s startup
    window, so the first run of a session after any git operation is lost -- and
    the retries are lost too, because each killed advisor orphans build children
    that hold the cache lock (see `kill_everything`).

    Paying for it here converts a mystery stall four runs deep into one visible,
    attributable wait before the ladder starts. Failure is reported, not raised:
    a broken build is worth saying out loud, but the advisor may still import a
    previously cached wheel, and refusing to climb over a probe is worse than
    trying.
    """
    say("pre-building the advisor wheel (uv run python -c 'import sts2_rs')")
    # Reap first: an orphan from a previous session holds the lock this build
    # needs, and the resulting failure looks like a slow build, not a deadlock.
    kill_everything()
    powershell(
        "Get-ChildItem -Path "
        "\"$env:LOCALAPPDATA\\uv\\cache\\sdists-v9\\editable\\*\\.lock\" "
        "-ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue"
    )
    started = time.time()
    try:
        proc = subprocess.run(
            ["uv", "run", "python", "-c", "import sts2_rs; print(sts2_rs.__file__)"],
            cwd=REPO / "sts2-pycli",
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=30 * 60,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        say(f"  advisor wheel did NOT build ({exc}); climbing anyway.")
        return False
    took = time.time() - started
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        say(f"  advisor wheel FAILED after {took:.0f}s: {' | '.join(tail)[:300]}")
        say("  the first run will probably stall in the startup window.")
        return False
    say(f"  advisor wheel ready in {took:.0f}s.")
    return True


def advisor_tail(lines: int = 8) -> str:
    """The most recent advisor launch's output, flattened onto one line.

    Scoped to the last `--- advisor launched` marker on purpose: the file is
    append-only across every launch of every run, so an unscoped tail shows a
    previous run's healthy advice while this one never started.
    """
    try:
        text = ADVISOR_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "no advisor_stdout.log"
    chunk = text.rsplit("--- advisor launched", 1)[-1]
    kept = [ln.strip() for ln in chunk.splitlines() if ln.strip()][-lines:]
    return " | ".join(kept)[-300:] if kept else "advisor logged nothing"


def steam_alive() -> bool:
    """Is the Steam client up AND logged in?

    Both halves are required. `ActiveUser` alone is not enough: the key keeps
    the last user id after the client dies, so it read as "logged in" for a
    full hour on 2026-08-07 while zero steam.exe processes existed. The process
    alone is not enough either -- a client still logging in cannot serve the
    game's Steamworks init.
    """
    out = powershell(
        "$p = @(Get-Process -Name steam -ErrorAction SilentlyContinue).Count; "
        "$u = (Get-ItemProperty 'HKCU:\\Software\\Valve\\Steam\\ActiveProcess' "
        "-ErrorAction SilentlyContinue).ActiveUser; "
        "if ($p -gt 0 -and $u -ne 0) { 'ready' } else { 'no' }"
    )
    return "ready" in (out or "")


def ensure_steam() -> bool:
    """Bring Steam up and logged in, and do not return until it is.

    The game is a Steam game: without a live client its Steamworks init fails,
    it sits on an error popup that never reaches the main menu, and the driver
    can only report the generic "no action committed" stall. Worse, on
    2026-08-07 a game launched against a half-ready client took the CLIENT down
    with it ~15 s later, so the next launch failed the same way -- a cycle that
    burned two runs and every retry in them.

    Started via WMI rather than Start-Process on purpose: a client started as a
    child of this driver's shell dies when that shell recycles.
    """
    if steam_alive():
        return True
    say("  Steam is not ready; starting it.")
    powershell(
        "Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        f"-Arguments @{{ CommandLine = '\"{STEAM_EXE}\"' }} | Out-Null"
    )
    deadline = time.monotonic() + STEAM_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if steam_alive():
            # Logged in is not the same as settled: the client needs a moment
            # before it can service an -applaunch.
            time.sleep(10)
            say("  Steam is up and logged in.")
            return True
        time.sleep(5)
    say("  Steam did NOT come up; the launch will almost certainly fail.")
    return False


def launch_game(stdout_log: Path) -> None:
    """Start the game THROUGH Steam and return immediately.

    Launching `SlayTheSpire2.exe` directly is what the driver used to do, and
    it works right up until Steam is not already up: the game then fails its
    Steamworks init AND the client shuts down ~15 s later, which poisons every
    subsequent launch (2026-08-07, issue 9). `-applaunch` makes Steam the
    launcher, so the ordering is Steam's problem rather than a race we lose.

    **A direct launch behind `ensure_steam()` was tried and reverted
    (2026-08-08).** The argument for it was that the readiness gate is the real
    cure and that `-applaunch` costs a visible foreground window plus Steam's
    `gameoverlayrenderer64.dll` in the present path, which matters on a box
    someone is streaming from. Both halves were then retracted in the same
    commit: `-WindowStyle Hidden` hides a console, not a Godot window, and
    Steam injects its overlay into any Steam game it detects regardless of who
    launched it. With neither benefit real, the deciding factor is mileage --
    `-applaunch` has ~40 unattended runs behind it and the direct path has the
    multi-hour outage of issue 9.

    Note `steam_appid.txt` already sits in the game directory and does NOT
    prevent this -- the client is the thing quitting, not the game.

    The stdout redirect is gone with the direct launch: Steam owns the child
    now, so the game's chatty stdout never reaches this driver's console
    anyway, which was the only reason to redirect it. The game writes its own
    `godot.log` regardless, and that is what every diagnosis actually reads.

    NOT via `powershell()`: that captures output through pipes, and the child
    inherits them, so the pipes only reach EOF when the GAME exits --
    `subprocess.run` then blocks for the entire run. The whole watch loop below
    silently never ran while a game was alive; every "the climb is stuck" on
    2026-08-06 was this one line. Handing the child DEVNULL is what makes the
    launch a launch.
    """
    ensure_steam()
    subprocess.run(
        [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            f"Start-Process -FilePath '{STEAM_EXE}' "
            f"-ArgumentList '-applaunch','{STEAM_APPID}'",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )


def write_conf(
    character: str,
    ascension: int,
    resume: bool,
    time_scale: float,
    advisor_brain: str = "search",
    seed: str | None = None,
) -> None:
    conf = {
        "port": 9002,
        "character": character,
        "ascension": ascension,
        "seed": seed,
        "num_runs": 1,
        "speed_patches": True,
        "agent_timeout_seconds": 0,
        "auto_start": True,
        "advisor_autoplay": True,
        # Never hand a decision back: nobody is at the keyboard to take it
        # (GH #874). advisor_autoplay already forces this; written anyway so
        # the conf says what the run does.
        "pause_on": "",
        "resume_run": resume,
        "fast_mode": "instant",
        "time_scale": time_scale,
        "client_run_config": False,
        "debug_player_hp": 0,
        "overlay": True,
        "stall_timeout_seconds": 300,
        # Cap the game's frame rate while the advisor thinks (SimThinkFps.cs)
        # -- the search runs in another process on this same laptop, and
        # rendering the idle board at full rate starved it ~5x.
        "think_fps": 10,
        "run_timeout_minutes": 360,
        # Which brain plays (PROTOCOL.md section 5, v5.2): "search" (MCTS +
        # the run net) or "instinct" (the e2e net, no search). Written
        # explicitly so the run never inherits whatever brain a human last
        # chose in the game's settings -- the conf key overrides that for
        # the session (SimConfig.AdvisorBrain).
        "advisor_brain": advisor_brain,
    }
    CONF.write_text(json.dumps(conf, indent=2), encoding="utf-8")


#: The advisor's own stdout, where the p_win / q values live. Cumulative and
#: OUTSIDE the repo, so a run's scores are otherwise interleaved with every
#: other run's and lost the moment anything truncates it. Moved from the mod
#: dir to %LOCALAPPDATA% with GH #380 (the mod dir is Program Files on a
#: default install; SimAdvisorProcess.AppDataDir is the C# twin).
ADVISOR_LOG = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "SpireMentor"
    / "advisor_stdout.log"
)
#: Where the mod rotates the above once it passes 10 MB
#: (SimAdvisorProcess.RotateOversizedLog). A climb that spans a rotation has
#: its slice split across the two files -- see `read_advisor_since`.
ADVISOR_LOG_ROTATED = ADVISOR_LOG.with_suffix(ADVISOR_LOG.suffix + ".1")
#: The game's own end-of-run record. Carries `seed`, `ascension`, and a
#: `map_point_history` with per-floor `player_stats` (deck / relics / hp /
#: potions) -- i.e. everything needed to set the situation up again. Searched
#: in EVERY modded profile (`history_files`): the slot the game plays is only
#: known once it has saved, after the "before" snapshot was taken.
HISTORY_GLOB = "*/modded/profile*/saves/history/*.run"


def advisor_log_offset() -> int:
    """Byte offset to slice this run's advisor output from, later."""
    try:
        return ADVISOR_LOG.stat().st_size
    except OSError:
        return 0


def read_advisor_since(offset: int) -> bytes:
    """This run's advisor output, from `offset` to now, across a rotation.

    The mod rotates the log to `.1` at 10 MB, which resets the offset this
    slice is taken from. Detect it (the live file is now SHORTER than the
    offset we recorded) and stitch: the tail of the rotated file from `offset`,
    then all of the fresh one. Without this the seek landed past EOF and the
    run archived a 0-byte advisor log -- silently, since the archive step
    reported success either way.

    The detection is a size comparison, so it needs the fresh file to still be
    shorter than `offset` when we read it. One run writes far less than the
    10 MB rotation threshold, so that holds; a run that somehow wrote 10 MB of
    its own after rotating would silently mis-slice again.
    """
    live_size = ADVISOR_LOG.stat().st_size if ADVISOR_LOG.exists() else 0
    if live_size >= offset:
        if live_size == 0:
            return b""
        with ADVISOR_LOG.open("rb") as fh:
            fh.seek(offset)
            return fh.read()
    # Rotated mid-run: the live file starts AFTER our offset, so all of it is
    # ours, and the bytes from `offset` up to the rotation are in the .1 file.
    live = ADVISOR_LOG.read_bytes() if live_size else b""
    old = b""
    if ADVISOR_LOG_ROTATED.exists():
        with ADVISOR_LOG_ROTATED.open("rb") as fh:
            fh.seek(offset)
            old = fh.read()
    say(
        f"  (advisor log rotated mid-run: {len(old)} B from "
        f"{ADVISOR_LOG_ROTATED.name} + {len(live)} B live)"
    )
    return old + live


def history_files() -> list[Path]:
    """Every run-history record of every modded profile."""
    try:
        return list((APPDATA / "SlayTheSpire2" / "steam").glob(HISTORY_GLOB))
    except OSError:
        return []


def history_snapshot() -> set[str]:
    """Paths of the run-history files that exist BEFORE this run."""
    return {str(p) for p in history_files()}


def archive_repro(
    tag: str, advisor_from: int, history_before: set[str]
) -> dict[str, object] | None:
    """Save everything needed to understand -- and re-create -- one run.

    Three artifacts beside the `godot.log` copy:

    * `<tag>.advisor.log` -- the slice of the advisor's stdout produced by THIS
      run. It holds the p_win/q values, and it is the piece that proved the
      Queen fight was evaluated as lost at move 1 (issue 10). It lives outside
      the repo and is append-only across every run (rotated at 10 MB, which
      `read_advisor_since` stitches back together), so without slicing it here
      the numbers survive only by luck.
    * `<tag>.run.json` -- the game's own history record, which is the actual
      repro key: it carries the **seed**, the ascension, and the per-floor
      `player_stats` (deck, relics, HP, potions).
    * `<tag>.current_run.save` -- the live save, when the run left one behind.
      A death usually clears it, so this is best-effort.

    Returns the run-history record itself if one could be read, for the ledger.
    That record -- not this driver's log scraping -- is the game's own verdict:
    it carries `win`, `ascension` (proof the requested level actually took) and
    `killed_by_encounter`. A run that ends without one was never graded by the
    game and must not be scored as a loss.
    """
    record: dict[str, object] | None = None
    try:
        blob = read_advisor_since(advisor_from)
        (OUT_DIR / f"{tag}.advisor.log").write_bytes(blob)
        if not blob:
            say("  (advisor log slice is EMPTY -- no p_win evidence archived)")
    except OSError as exc:
        say(f"  (could not slice the advisor log: {exc})")

    try:
        fresh = sorted(
            (p for p in history_files() if str(p) not in history_before),
            key=lambda p: p.stat().st_mtime,
        )
        if fresh:
            newest = fresh[-1]
            shutil.copy2(newest, OUT_DIR / f"{tag}.run.json")
            try:
                parsed = json.loads(newest.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    record = parsed
            except (json.JSONDecodeError, OSError):
                pass
    except OSError as exc:
        say(f"  (could not archive the run history: {exc})")

    try:
        live = PROGRESS.parent / "current_run.save"
        if live.exists():
            shutil.copy2(live, OUT_DIR / f"{tag}.current_run.save")
    except OSError:
        pass

    return record


def archive_log(dest: Path) -> None:
    """Keep the diagnostically useful lines, never the whole file.

    A wedged run's `godot.log` is gigabytes of one repeated renderer error. The
    mod's own lines and the game's `[INFO]`s are the part anyone reads, and a
    few hundred errors are plenty to identify the flood, so the archive is the
    filtered view -- small enough to commit next to a writeup.
    """
    try:
        size = GODOT_LOG.stat().st_size
    except OSError:
        return
    if size <= 25 * 1024 * 1024:
        shutil.copy2(GODOT_LOG, dest)
        return
    errors = 0
    with GODOT_LOG.open(encoding="utf-8", errors="replace") as src, dest.open(
        "w", encoding="utf-8"
    ) as out:
        out.write(f"# EXTRACT of a {size / 1e9:.2f} GB godot.log (flood-filtered)\n")
        for line in src:
            if _is_mod_line(line) or line.startswith("[INFO]"):
                out.write(line)
            elif "ERROR" in line and errors < 400:
                errors += 1
                out.write(line)


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------


@dataclass
class RunResult:
    character: str
    ascension: int
    verdict: str  # win | loss | stall | crash | timeout
    detail: str
    floors: int
    actions: int
    refusals: int
    minutes: float
    log: str
    ascension_after: int
    # Recorded so a crash or a refusal spike can be attributed to the speed it
    # ran at -- issue 15 was only diagnosable because the value was known.
    time_scale: float = 1.0
    #: The brain that played ("search" / "instinct"): a win rate means nothing
    #: without it once both can climb.
    advisor_brain: str = "search"
    #: The game's own run seed, from the archived history record. The ledger
    #: had no seed at all before, so a run worth re-creating could not be
    #: found again even though the game had written one down.
    seed: str | None = None
    #: The game's own verdict, from that same record: `win`, the ascension the
    #: run ACTUALLY ran at, and what killed it. `record_win is None` means the
    #: game never wrote a record -- an ungraded run, which a win-rate scorer
    #: must exclude rather than count as a loss. `record_ascension` is the only
    #: proof that a requested level above the profile's max took effect.
    record_win: bool | None = None
    record_ascension: int | None = None
    killed_by: str | None = None
    run_seconds: int | None = None


def read_log() -> str:
    """The parts of `godot.log` this driver reasons about, streamed.

    Never `read_text()`: a wedged renderer turns this file into gigabytes of
    one repeated error, and every signal this driver looks for lives on a
    `[SpireMentor]` or `[INFO]` line. Raw `ERROR:` lines are kept only up to a
    cap -- enough to name the flood, not enough to hold it in memory.
    """
    kept: list[str] = []
    errors = 0
    try:
        with GODOT_LOG.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if _is_mod_line(line) or line.startswith("[INFO]"):
                    kept.append(line)
                elif line.startswith("ERROR") and errors < 2000:
                    errors += 1
                    kept.append(line)
    except OSError:
        return ""
    return "".join(kept)


def log_signature() -> tuple[int, int] | None:
    """Identity of the `godot.log` on disk right now, or None if there is none.

    Godot rotates its log at startup -- the old one becomes
    `godot<timestamp>.log` and a fresh `godot.log` is created -- so an
    UNCHANGED signature across a launch means the launch never happened.
    """
    try:
        st = GODOT_LOG.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def watch_one_attempt(
    run_started: float, stale_sig: tuple[int, int] | None = None
) -> tuple[str, str]:
    """Poll until this game process reaches a terminal state.

    Returns (verdict, detail). `win`/`loss` end the run; `stall`/`crash`/
    `timeout` are recoverable -- the caller comes back for the same run.

    The startup window is measured from THIS launch while the hard cap is
    measured from the run: a resumed attempt starts with an hour already on the
    run clock, and charging that to its startup window would declare every
    resume dead on its first poll.
    """
    attempt_started = time.time()
    last_progress = -1
    last_move = time.time()
    size_at_last_move = 0
    # "the game died" and "the game never came up" reach the crash check as the
    # same absent process, ~100 s in either way -- and that is also issue 0's
    # signature, so an undifferentiated message re-opens a closed bug. Latch the
    # first sighting so the two can be told apart, and give a launch that has
    # simply not appeared yet one more grace window (slow launches are expected
    # on this box: see docs/experiments/2026-08-07-host-unresponsiveness/).
    ever_alive = False
    appear_grace = GAME_APPEAR_GRACE_S

    while True:
        time.sleep(POLL_SECONDS)
        # A LOG THAT DID NOT ROTATE IS THE PREVIOUS SESSION'S LOG. Reading it
        # anyway is not a harmless no-op: every regex below then matches the
        # OLD run, and `VICTORY_RE` matches on the very first poll, so the
        # driver scores a phantom win ~84 s after a launch that never happened.
        #
        # Not hypothetical. 2026-08-25, sweep `logs/f7-mirror-check16`: the
        # Steam account was in-game on another machine, so `-applaunch` did
        # nothing, and 17 consecutive "runs" were graded
        # `win -- has won a run with ancient [49 floors, 1134 actions]` off one
        # 16:04 log -- all 17 archived copies byte-identical at 470,734 bytes,
        # all 17 advisor slices empty. Only the profile-authority grader
        # (`record_win=None` -> UNGRADED) kept them out of the ledger, and
        # because ungraded runs are replayed rather than scored, the sweep was
        # on course to do that 1000 times.
        if log_signature() == stale_sig:
            if time.time() - attempt_started <= appear_grace:
                continue
            if appear_grace == GAME_APPEAR_GRACE_S:
                appear_grace *= 2
                say("godot.log has not rotated yet; waiting once more")
                continue
            return "stall", (
                "godot.log never rotated -- the game did not launch at all "
                "(Steam busy on another machine? another copy already "
                "running?). Refusing to grade the previous session's log."
            )
        text = read_log()
        actions = text.count(TAKEOVER)
        floors = text.count("Takeover executed on NMapScreen:")
        # Questions the GAME raised, whether or not anything answered them --
        # the discriminator for the startup stall below.
        decisions = text.count("Decision up:")
        # An enqueued action is not progress. A blocked action queue -- the
        # game waiting on a selection nothing describes -- makes the advisor
        # re-request forever, and every pass logs another `Takeover executed`:
        # 2,795 of them in 2h40m read as a perfectly healthy run
        # (2026-08-06). Progress is actions the BOARD accepted, so subtract
        # the ones the mod itself reports as having changed nothing.
        refused = len(REFUSED_RE.findall(text))
        progress = actions - refused
        elapsed = time.time() - attempt_started
        # Every poll, unconditionally. "The climb is stuck" and "the watchdog
        # is stuck" look identical from outside, and this file is the only
        # thing that tells them apart -- it is written before any probe that
        # could block, so a heartbeat that stops names the line that hung.
        try:
            with (OUT_DIR / "heartbeat.log").open("a", encoding="utf-8") as fh:
                fh.write(
                    f"{now()} floors={floors} actions={actions} refused={refused} "
                    f"idle={int(time.time() - last_move)}s size={size_at_last_move}\n"
                )
        except OSError:
            pass
        try:
            size = GODOT_LOG.stat().st_size
        except OSError:
            size = 0

        if progress != last_progress:
            last_progress, last_move, size_at_last_move = progress, time.time(), size

        run_text = this_run(text)
        if VICTORY_RE.search(run_text):
            return "win", "log: has won a run with ancient"
        if GAMEOVER_RE.search(run_text):
            lost = re.search(r"has lost to encounter.*", run_text)
            return "loss", (lost.group(0)[:120] if lost else "game over screen")
        if RUNAWAY_RE.search(text):
            m = RUNAWAY_RE.search(text)
            return "stall", f"runaway guard: {text[m.start():m.start() + 140]}"
        if time.time() - attempt_started > appear_grace:
            alive = game_alive()
            ever_alive = ever_alive or alive
            if not alive:
                if ever_alive:
                    return "crash", f"game died after {floors} floors"
                if appear_grace == GAME_APPEAR_GRACE_S:
                    appear_grace *= 2
                    say(f"game not up after {GAME_APPEAR_GRACE_S}s; waiting once more")
                else:
                    return "crash", f"game never appeared within {appear_grace:.0f}s"
        if size - size_at_last_move > FLOOD_BYTES:
            top = ""
            tail = text[-200000:]
            hits = re.findall(r"^ERROR: (.*)$", tail, re.M)
            if hits:
                top = max(set(hits), key=hits.count)[:100]
            return "stall", (
                f"log flooded {(size - size_at_last_move) / 1e6:.0f} MB with no "
                f"action at floor {floors}: {top}"
            )
        if actions == 0 and elapsed > STARTUP_SECONDS:
            # Three unrelated failures used to share this one message and each
            # needs a different cure: a game that never reached the menu (issue
            # 9, Steam), a game wedged before it could ask anything (issue 2),
            # and an advisor that never came up (issue 8, the uv build). The
            # game's own decision count splits the first two from the third,
            # which is the split that was actually missing -- "the advisor is
            # still building" spent four runs disguised as a game stall.
            if decisions == 0:
                return "stall", (
                    "no decision raised in the startup window -- the GAME never "
                    "asked (launch/menu/wedge); the advisor is not implicated"
                )
            return "stall", (
                f"the ADVISOR never answered: {decisions} decisions raised, 0 "
                f"actions, in {STARTUP_SECONDS}s. advisor tail: {advisor_tail()}"
            )
        if progress > 0 and time.time() - last_move > STALL_SECONDS:
            hits = re.findall(r"\[SimAdvisor\] Decision up: (.*)", text)
            where = hits[-1][:120] if hits else "?"
            if refused > 5:
                return "stall", (
                    f"livelock: {refused} actions the board refused, "
                    f"none accepted for {STALL_SECONDS}s, at: {where}"
                )
            return "stall", f"no action for {STALL_SECONDS}s at: {where}"
        # Per ATTEMPT, not per run. Measured from the run, a resumed attempt
        # inherits a clock that is already past the cap and times out on its
        # first poll -- run 002 burned all four attempts in 75 seconds that
        # way. The run as a whole is bounded by RUN_BUDGET_SECONDS below.
        if elapsed > RUN_CAP_SECONDS:
            return "timeout", f"attempt exceeded {RUN_CAP_SECONDS // 3600}h"
        if time.time() - run_started > RUN_BUDGET_SECONDS:
            return "timeout", f"run exceeded its {RUN_BUDGET_SECONDS // 3600}h budget"


def play_one(
    character: str,
    ascension: int,
    index: int,
    resume_first: bool = False,
    time_scale: float = 1.0,
    advisor_brain: str = "search",
    seed: str | None = None,
) -> RunResult:
    global PROGRESS
    stamp = datetime.now().strftime("%m%d-%H%M")
    tag = f"{index:03d}-{character.lower()}-a{ascension}-{stamp}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every profile's, because which one the game plays is only known from the
    # log once it has saved (`profile_in_log`).
    before_by_profile: dict[Path, int] = {}
    for save in {PROGRESS, *all_progress_saves()}:
        try:
            before_by_profile[save] = ladder(save)[character]
        except RuntimeError:
            pass
    before = before_by_profile.get(PROGRESS, 0)
    started = time.time()
    verdict = detail = ""
    # Captured BEFORE the first launch so the archived advisor slice covers
    # every attempt of this run, retries included -- a run that only got
    # interesting on its third resume would otherwise lose the first two.
    advisor_from = advisor_log_offset()
    history_before = history_snapshot()

    for attempt in range(1, RESUME_ATTEMPTS + 2):
        kill_everything()
        resume = attempt > 1 or resume_first
        write_conf(character, ascension, resume, time_scale, advisor_brain, seed)
        how = "resuming the saved run" if resume else f"ascension {ascension} (max {before})"
        say(f"RUN {tag}: launching {character}, {how} [attempt {attempt}]")
        # Straight to a scratch path: the game's stderr is the same flood as
        # godot.log and reached 1.3 GB once. Nothing reads it that godot.log
        # does not already carry.
        # Read AFTER `kill_everything()` and immediately before the launch, so
        # it is the identity of the file this launch has to replace.
        stale_sig = log_signature()
        launch_game(Path(os.environ.get("TEMP", "/tmp")) / "sts2_climb_stdout.log")

        verdict, detail = watch_one_attempt(started, stale_sig)
        if verdict in ("win", "loss"):
            break
        if attempt <= RESUME_ATTEMPTS:
            say(f"RUN {tag}: {verdict} -- {detail}; killing and resuming the saved run.")
        else:
            say(f"RUN {tag}: {verdict} after {RESUME_ATTEMPTS} resumes; giving up on this run.")

    live = profile_in_log(read_log())
    if live is not None and live != PROGRESS:
        say(
            f"  the game played {live.parent.parent.name}, not "
            f"{PROGRESS.parent.parent.name}; grading and climbing that profile from now on."
        )
        PROGRESS = live
        before = before_by_profile.get(live, before)

    # The save is written when the run resolves; give it a moment, and grade by
    # the profile rather than by what the log claimed.
    #
    # A new run-history record ends the wait too. Ladder movement alone cannot:
    # a run played BELOW the character's max ascension (every run of a
    # fixed-`--ascension` win-rate sweep) never increments anything, so this
    # loop used to burn its full 60 s after every single run.
    after = before
    for _ in range(15):
        time.sleep(4)
        after = ladder()[character]
        if after != before or history_snapshot() - history_before:
            break

    text = read_log()
    result = RunResult(
        character=character,
        ascension=ascension,
        verdict=verdict,
        detail=detail,
        floors=text.count("Takeover executed on NMapScreen:"),
        actions=text.count(TAKEOVER),
        refusals=len(
            re.findall(r"changed nothing on the board|the game refuses", text)
        ),
        minutes=round((time.time() - started) / 60, 1),
        log=f"{tag}.godot.log",
        ascension_after=after,
        time_scale=time_scale,
        advisor_brain=advisor_brain,
    )
    kill_everything()
    archive_log(OUT_DIR / result.log)
    # After kill_everything(): the game writes its history record when the run
    # resolves, and the ladder wait above already gave it time to land.
    record = archive_repro(tag, advisor_from, history_before) or {}
    result.seed = str(record["seed"]) if record.get("seed") is not None else None
    win = record.get("win")
    result.record_win = bool(win) if isinstance(win, bool) else None
    asc = record.get("ascension")
    result.record_ascension = asc if isinstance(asc, int) else None
    killed = record.get("killed_by_encounter")
    result.killed_by = str(killed) if killed else None
    secs = record.get("run_time")
    result.run_seconds = secs if isinstance(secs, int) else None
    # The two graders disagreeing is a defect signal, not a rounding error --
    # a watchdog-killed VICTORY reading as a loss is exactly how live wins were
    # lost before (2026-08-03). Say so loudly; the record wins.
    if result.record_win is not None and verdict in ("win", "loss"):
        if result.record_win != (verdict == "win"):
            say(
                f"  GRADER MISMATCH: log said {verdict}, the game's own record "
                f"says win={result.record_win}. Trusting the record."
            )
    if result.record_ascension is not None and result.record_ascension != ascension:
        say(
            f"  ASCENSION MISMATCH: asked for A{ascension}, the run actually ran "
            f"at A{result.record_ascension}."
        )

    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"finished": now(), **result.__dict__}) + "\n")
    say(
        f"RUN {tag}: {result.verdict} -- {result.detail} "
        f"[{result.floors} floors, {result.actions} actions, "
        f"{result.refusals} refusals, {result.minutes} min, "
        f"ascension {before} -> {after}, record win={result.record_win} "
        f"a{result.record_ascension} seed={result.seed}]"
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="print the ladder and exit")
    ap.add_argument("--once", action="store_true", help="play a single run")
    ap.add_argument("--character", help="climb only this character")
    ap.add_argument("--max-runs", type=int, default=1000)
    ap.add_argument(
        "--ascension",
        type=int,
        default=None,
        metavar="N",
        help=(
            "play every run at this FIXED ascension instead of climbing, and "
            "stop after --graded-runs runs the game itself graded (win or "
            "loss). This is the win-rate mode; it needs --character. The level "
            "may exceed the character's unlocked max -- "
            "NAscensionPanel.SetAscensionLevel does not clamp and neither does "
            "StartRunLobby.SyncAscensionChange (v0.107.1), and every run's "
            "archived history record carries the ascension it truly ran at, so "
            "a silent clamp would be visible rather than assumed. Nothing is "
            "disturbed on the ladder either: MaxAscension only increments on a "
            "win AT the current max."
        ),
    )
    ap.add_argument(
        "--graded-runs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "target number of GRADED runs for --ascension mode (default: run "
            "--max-runs times whatever happens). A stall/crash/timeout run is "
            "replayed rather than scored -- counting an ungraded run as a loss "
            "biases the very number this mode exists to measure."
        ),
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        metavar="PATH",
        help=(
            "where the ledger, logs and per-run artifacts go (default "
            "logs/ascension_climb). Give a sweep its own directory so its "
            "ledger is the sweep, not the climb history plus the sweep."
        ),
    )
    ap.add_argument(
        "--cooldown",
        type=int,
        default=90,
        metavar="SECONDS",
        help=(
            "idle seconds between runs, with the game closed (default 90; 0 to "
            "disable). This box is a laptop whose CPU package, dGPU and the "
            "QuickSync encoder driving the Moonlight stream all share one "
            "cooling loop, and back-to-back runs at 99%% duty pin SW Power Cap "
            "and SW Thermal Slowdown until single-threaded throughput drops 5x "
            "and the stream becomes unusable -- see "
            "docs/experiments/2026-08-07-host-unresponsiveness/. Heat sheds at "
            "only ~3 C per 25 min here, so this does not fully clear it; it "
            "just stops the ratchet. Costs ~7%% throughput at a 20-minute "
            "median run."
        ),
    )
    ap.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        metavar="X",
        help=(
            "whole-game speed (Engine.TimeScale), 0.1-4.0, default 1.0 = "
            "untouched. Worth raising: ~63%% of a run's wall-clock is the game "
            "animating, not MCTS thinking (37%%, measured on runs 032/033), and "
            "the search runs in another process this cannot reach -- so the "
            "ceiling is ~1.9x however high you go. It stacks with the Autopilot "
            "multipliers (waits x0.1, Spine x5), so at 4.0 the game's internal "
            "settle-waits run ~40x shorter than vanilla; run 046 crashed in the "
            "Spine animation path at a hand-set 4.0 (issue 15, n=1). The value "
            "is recorded per run in climb.jsonl -- watch refusals as well as "
            "crashes, and back off if either climbs."
        ),
    )
    ap.add_argument(
        "--advisor-brain",
        choices=("search", "instinct"),
        default="search",
        help=(
            "which brain plays: search (MCTS for the fights, the run net for "
            "the screens; default) or instinct (the end-to-end net, no search "
            "-- Ironclad on v0.111.0 only; elsewhere the advisor falls back "
            "to the search and says so in its note). Written to the conf as "
            "advisor_brain and recorded per run in climb.jsonl."
        ),
    )
    ap.add_argument(
        "--seed",
        default=None,
        help=(
            "play this run seed (the conf's seed; e.g. one from a ledger line, "
            "to replay a run after a fix). Default: a random seed."
        ),
    )
    ap.add_argument(
        "--skip-prebuild",
        action="store_true",
        help=(
            "do not build the advisor wheel before climbing. The build is "
            "normally free (a warm cache imports in under a second) and saves "
            "the run that would otherwise discover a cold one 300 s into its "
            "startup window -- skip it only when something else is already "
            "holding uv."
        ),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "continue the run already saved instead of starting a new one -- "
            "use when restarting the driver mid-run so the restart costs "
            "nothing. Safe unconditionally: the mod falls back to a new run "
            "when there is nothing to continue."
        ),
    )
    args = ap.parse_args()

    if args.out_dir:
        global OUT_DIR, LEDGER
        OUT_DIR = Path(args.out_dir)
        LEDGER = OUT_DIR / "climb.jsonl"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.ascension is not None and not args.character:
        ap.error("--ascension needs --character (it fixes both, and grades both)")

    state = ladder()
    if args.status:
        print(f"profile in play: {PROGRESS.parent.parent.name}  ({PROGRESS})")
        for name in CLIMB_ORDER:
            print(f"{name:12s} A{state[name]}/{MAX_ASCENSION}")
        print(f"wins still needed: {wins_remaining(state)}")
        return 0

    say(f"profile in play: {PROGRESS.parent.parent.name} ({PROGRESS}); ladder {state}")

    if not args.skip_prebuild:
        ensure_advisor_wheel()

    index = sum(1 for _ in LEDGER.open(encoding="utf-8")) if LEDGER.exists() else 0
    played = 0
    graded = wins = 0
    while played < args.max_runs:
        if args.ascension is None:
            state = ladder()
            target = next_target(state, args.character)
            if target is None:
                say(f"LADDER COMPLETE: {state}")
                return 0
            level = state[target]
        else:
            target, level = args.character, args.ascension
            if args.graded_runs is not None and graded >= args.graded_runs:
                break
        index += 1
        played += 1
        # Only the first run of this process can inherit a saved run; after
        # that a resume would pick up the run this loop just finished.
        result = play_one(
            target,
            level,
            index,
            resume_first=args.resume and played == 1,
            time_scale=args.time_scale,
            advisor_brain=args.advisor_brain,
            seed=args.seed,
        )
        if args.ascension is not None:
            if result.record_win is None:
                say(f"  UNGRADED ({result.verdict}) -- replaying, not scoring it.")
            else:
                graded += 1
                wins += 1 if result.record_win else 0
                target_note = f"/{args.graded_runs}" if args.graded_runs else ""
                say(
                    f"  SWEEP {target} A{level}: {wins}/{graded}{target_note} graded "
                    f"= {100 * wins / graded:.1f}% ({played} runs played)"
                )
        if args.once:
            break
        # Let the chassis breathe before the next launch. play_one() has already
        # killed the game, so this is genuine idle time, not a slower run.
        if args.cooldown > 0:
            say(f"cooldown {args.cooldown}s (thermal; --cooldown 0 to disable)")
            time.sleep(args.cooldown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
