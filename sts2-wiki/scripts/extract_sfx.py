#!/usr/bin/env python3
"""Extract the game's SFX banks from the Godot PCK, decode them to named WAVs.

Music is NOT extracted. The game keeps music and ambience in separate FMOD
banks — seven music banks (`act1_a1`, `act1_a2`, `act1_b1`, `act2_a1`,
`act2_a2`, `act3_a1`, `act3_a2`) plus `ambience`, 269.7 MB together — so
excluding it is an allowlist of the four SFX-side banks rather than a content
judgement.

Rendering (`render_events`) decodes each named event in the extracted banks
to its own WAV via the real FMOD Studio engine: on Linux the FMOD Studio API
2.03.14 SDK (`FMOD_SDK_ROOT`, not committed); on Windows the SDK too when
set, else the `fmod.dll`/`fmodstudio.dll` the game installs (`FMOD_LIB_DIR`,
defaulting to the pck's directory). `find_fmod_runtime` holds the rules.
`Master.bank`'s two `event:/music/`
entries are excluded by `is_music` as a second layer on top of the bank
allowlist, since Master.bank itself is required (it carries the mixer); see
`is_music` for the one further path it denylists by exact match.
"""

from __future__ import annotations

import ctypes
import os
import re
import struct
import time
from pathlib import Path
from typing import NamedTuple

FMOD_OK = 0
FMOD_STUDIO_LOAD_BANK_NORMAL = 0
FMOD_STUDIO_INIT_NORMAL = 0
# Studio normally processes commands and loads sample data on threads of its
# own, so how far into an NRT render an event's audio arrives depends on
# wall-clock scheduling: renders of one event came out with different leading
# delays, lengths, and once silence. Driving both from `Update()` (plus the
# explicit sample preload in `_render_one`) makes a render a pure function of
# the banks, the FMOD build and `RANDOM_SEED`.
FMOD_STUDIO_INIT_SYNCHRONOUS_UPDATE = 0x00000004
FMOD_STUDIO_INIT_LOAD_FROM_UPDATE = 0x00000010
FMOD_INIT_NORMAL = 0
# `FMOD_INIT_STREAM_FROM_UPDATE` (fmod_common.h, 0x1): "No stream thread is
# created internally. Streams are driven from System::update."
#
# LOAD-BEARING for this renderer, and its absence was the single biggest audio
# bug the workbench has had. The NRT WAV writer advances the mixer once per
# `Update()` call, i.e. hundreds of times faster than real time — but the
# stream thread refills its buffer in WALL-CLOCK time. Every event backed by a
# streaming sound therefore replayed the same buffer over and over: FMOD's
# default stream buffer is 400 ms, which is exactly the period MEASURED in the
# broken renders (`gold_2`, `regent_greeting`, `silent_die`, … all repeating on
# a 0.400 s cycle, all reporting PLAYBACK_STOPPED never).
#
# It hit 135 of the 566 events, i.e. every `_die`, every `_select`, every NPC
# voice line, all three gold cues and `temp/sfx/game_over`. Downstream that
# came out as 18 shipped files of 8 s of stuttering loop and 91 shipped files
# truncated to a single 0.408 s repetition by `encode_sfx.detect_loop` — sounds
# that are not in the game either way. With the flag set, every one of those
# events reports STOPPED on its own and renders its real length.
#
# NOT a substitute for CAP_SECONDS: a genuinely looping event (the ambience/
# `*_loop` family) still has to be capped. The cap is the backstop; this is the
# correctness fix.
FMOD_INIT_STREAM_FROM_UPDATE = 0x00000001
FMOD_STUDIO_PLAYBACK_STOPPED = 2
# Non-realtime, so a render finishes as fast as Update() is called rather than
# in wall-clock time. Confirmed against api/core/inc/fmod_common.h for 2.03.14.
FMOD_OUTPUTTYPE_WAVWRITER_NRT = 5
# `FMOD_RESULT` ordinal 74, counted off `api/core/inc/fmod_common.h` for
# 2.03.14. `SetParameterByName` returns this — not a hard failure — when the
# event simply has no parameter by that name, which is the normal case for
# every event but the impact family. See `RENDER_PARAMS`.
FMOD_ERR_EVENT_NOTFOUND = 74

# Parameter values to apply before rendering, so an event that the game drives
# with a parameter is captured the way the game actually sounds rather than at
# its authored default.
#
# `EnemyImpact_Intensity` is the only one that matters, and it is the ONLY
# one-shot parameter the v0.107.1 decompilation ever sets:
# `SfxCmd.PlayDamage` (SfxCmd.cs:82) plays every monster's `TakeDamageSfx`
# with `{"EnemyImpact_Intensity", 2f}`, i.e. EVERY hit that lands in the game.
# Rendering those events at the default (0) captured a different layer of the
# event than any player has ever heard. The decomp's other `SetParam` calls
# (CubexConstruct's `loop`/`enemy_hurt`, WaterfallGiant's
# `waterfall_giant_sfx`, and `NCreature._sfxLoops`' stop-parameter) all drive
# LOOPS, which this renderer excludes anyway.
#
# Applied best-effort to every event: an event with no such parameter answers
# FMOD_ERR_EVENT_NOTFOUND and is left alone. That is safe here precisely
# because the name is namespaced to the impact family — no other event in
# these four banks carries it.
RENDER_PARAMS: dict[str, float] = {"EnemyImpact_Intensity": 2.0}

SFX_BANKS: tuple[str, ...] = (
    "banks/desktop/Master.bank",
    "banks/desktop/Master.strings.bank",
    "banks/desktop/sfx.bank",
    "banks/desktop/temp_sfx.bank",
)


def _read_pck_directory(f) -> list[tuple[str, int, int]]:
    """Return (path, offset, size) for every entry, offsets already absolute."""
    magic = f.read(4)
    if magic != b"GDPC":
        raise ValueError(f"not a Godot pck: {magic!r}")
    fmt_version = struct.unpack("<I", f.read(4))[0]
    f.read(12)  # engine major/minor/patch
    f.read(4)  # pack flags
    file_base = struct.unpack("<Q", f.read(8))[0]
    directory_offset = 0
    if fmt_version >= 3:
        directory_offset = struct.unpack("<Q", f.read(8))[0]
    f.read(64)  # reserved
    if fmt_version >= 3:
        f.seek(directory_offset)
    count = struct.unpack("<I", f.read(4))[0]
    entries: list[tuple[str, int, int]] = []
    for _ in range(count):
        path_len = struct.unpack("<I", f.read(4))[0]
        path = f.read(path_len).rstrip(b"\x00").decode("utf-8")
        offset = struct.unpack("<q", f.read(8))[0]
        size = struct.unpack("<q", f.read(8))[0]
        f.read(16)  # md5
        f.read(4)  # flags
        entries.append((path.removeprefix("res://"), file_base + offset, size))
    return entries


def extract_banks(pck_path: Path, out_dir: Path) -> list[Path]:
    """Write the four SFX banks to *out_dir*; return the paths written.

    Raises ValueError if the file is not a pck or does not hold all four banks.
    """
    wanted = set(SFX_BANKS)
    written: list[Path] = []
    with open(pck_path, "rb") as f:
        entries = _read_pck_directory(f)
        for path, offset, size in entries:
            if path not in wanted:
                continue
            dest = out_dir / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            f.seek(offset)
            dest.write_bytes(f.read(size))
            written.append(dest)
    missing = wanted - {p.relative_to(out_dir).as_posix() for p in written}
    if missing:
        raise ValueError(f"pck is missing expected banks: {sorted(missing)}")
    return written


def event_to_relpath(event: str) -> str:
    """`event:/sfx/block_gain` -> `sfx/block_gain` (no extension)."""
    if not event.startswith("event:/"):
        raise ValueError(f"not an FMOD event path: {event!r}")
    return event.removeprefix("event:/")


SILENTLY_EXCLUDED_EVENTS = frozenset(
    {
        # Lives in temp_sfx.bank (allowlisted) under event:/temp/, not
        # event:/music/, so the namespace check below misses it. Confirmed
        # via ffprobe: 85.7s and never reports FMOD_STUDIO_PLAYBACK_STOPPED
        # within render's 4000-Update()/~30s cap (i.e. it loops indefinitely)
        # — a duration and looping behaviour no other SFX event in these
        # banks exhibits, and consistent with "temp_music" naming a
        # placeholder background-music track rather than a sound effect.
        # Denylisted by exact path pending confirmation from the game data
        # owner; flagged in the Task 2 report rather than shipped silently.
        "event:/temp/temp_music",
    }
)


def is_music(event: str) -> bool:
    """True for events that must never be rendered as game SFX.

    The bank allowlist is not sufficient on its own: `Master.bank` is
    required (it carries the mixer) and it ships two menu-music events
    (`event:/music/menu/menu_1`, `event:/music/menu_update` — its third
    "event" per `GetEventCount` is `snapshot:/pause`, a mixer snapshot with
    no audio, dropped separately in `_list_event_paths`). Excluding the
    act/ambience banks keeps the soundtrack out; this keeps the menu music
    out too, so "no music" holds at the event level as well as the bank
    level. `SILENTLY_EXCLUDED_EVENTS` additionally denylists one specific
    non-`/music/`-namespace event that content-inspection shows is music in
    substance if not in path — see its comment above.
    """
    return event.startswith("event:/music/") or event in SILENTLY_EXCLUDED_EVENTS


# The Windows game install ships the FMOD runtime it was built against
# (`fmod.dll` + `fmodstudio.dll`, 2.03.06 in v0.107.1) next to
# SlayTheSpire2.exe. On Windows that is the whole dependency: no SDK download.
GAME_FMOD_DLLS = ("fmod.dll", "fmodstudio.dll")

# Where the FMOD Studio API keeps its core and studio libraries, per platform.
_SDK_LIBS = {
    "nt": ("api/core/lib/x64/fmod.dll", "api/studio/lib/x64/fmodstudio.dll"),
    "posix": ("api/core/lib/x86_64/libfmod.so", "api/studio/lib/x86_64/libfmodstudio.so"),
}


class FmodRuntime(NamedTuple):
    """The core + studio libraries `render_events` loads.

    `header` is the SDK's `fmod_common.h`; the game's DLLs come without
    headers (None), so their version is read from the DLL itself.
    """

    core: Path
    studio: Path
    header: Path | None


def find_fmod_runtime(default_lib_dir: Path | None = None) -> FmodRuntime | None:
    """The FMOD runtime to render with, or None when there is none.

    The single source of these rules (scripts/import_game.py asks here too):
    1. `FMOD_SDK_ROOT`, the FMOD Studio API, in this platform's layout.
    2. Windows only: the game's own DLLs in `FMOD_LIB_DIR`, else in
       *default_lib_dir* (the folder holding the pck). They are Windows
       binaries; Linux cannot load them even when Steam installed the game
       through Proton.
    An SDK that lacks this platform's libraries (a Windows SDK set on Linux,
    or the other way round) falls through to the next rule.
    """
    root = os.environ.get("FMOD_SDK_ROOT")
    if root:
        core, studio = (Path(root) / rel for rel in _SDK_LIBS[os.name])
        if core.is_file() and studio.is_file():
            return FmodRuntime(core, studio, Path(root) / "api/core/inc/fmod_common.h")
    if os.name == "nt":
        lib_dir = os.environ.get("FMOD_LIB_DIR") or default_lib_dir
        if lib_dir:
            core, studio = (Path(lib_dir) / n for n in GAME_FMOD_DLLS)
            if core.is_file() and studio.is_file():
                return FmodRuntime(core, studio, None)
    return None


def _fmod_libs(runtime: FmodRuntime) -> tuple[ctypes.CDLL, ctypes.CDLL, int]:
    """Load the core and studio libraries; also return their header version.

    Core first, deliberately: libfmodstudio.so links against libfmod.so, so the
    core library must already be resident when the studio one loads.
    """
    core = ctypes.CDLL(str(runtime.core))
    studio = ctypes.CDLL(str(runtime.studio))
    return core, studio, _header_version(runtime)


def _dll_version(dll: Path) -> tuple[int, int, int]:
    """(major, minor, patch) from a DLL's VS_FIXEDFILEINFO (Windows only).

    FMOD stamps its release into the low three parts of the four-part file
    version: `fmod.dll` 0.2.3.6 (shown as "2.3.6") is the 2.03.06 runtime.
    """
    ver = ctypes.WinDLL("version")  # type: ignore[attr-defined]
    size = ver.GetFileVersionInfoSizeW(str(dll), None)
    if not size:
        raise SystemExit(f"{dll} carries no version resource")
    buf = ctypes.create_string_buffer(size)
    if not ver.GetFileVersionInfoW(str(dll), 0, size, buf):
        raise SystemExit(f"could not read the version resource of {dll}")
    info = ctypes.c_void_p()
    length = ctypes.c_uint()
    if not ver.VerQueryValueW(buf, "\\", ctypes.byref(info), ctypes.byref(length)):
        raise SystemExit(f"could not read the fixed file info of {dll}")
    # VS_FIXEDFILEINFO: signature, struct version, then FileVersionMS/LS.
    fields = ctypes.cast(info, ctypes.POINTER(ctypes.c_uint32 * 4)).contents
    ms, ls = fields[2], fields[3]
    return ms & 0xFFFF, ls >> 16, ls & 0xFFFF


def fmod_header_version(major: int, minor: int, patch: int) -> int:
    """FMOD's `FMOD_VERSION` encoding: 2.03.06 -> 0x00020306 (the decimal
    digits of each part, read as hex)."""
    return int(f"{major}{minor:02d}{patch:02d}", 16)


def _check(result: int) -> None:
    if result != FMOD_OK:
        raise RuntimeError(f"FMOD error {result}")


# Seed for FMOD's random number generators. Nearly every event randomises
# (which round-robin take of each layer, pitch, volume): unseeded, two renders
# of the same event are different audio, so every import shipped a different
# sound set. Seeded, each event renders the same take every time on a given
# FMOD build. Any nonzero value; 0 is FMOD's "seed from the clock" default.
RANDOM_SEED = 0x53545332  # "STS2"


class FmodAdvancedSettings(ctypes.Structure):
    """`FMOD_ADVANCEDSETTINGS` (fmod_common.h) up to `maxConvolutionThreads`.

    FMOD versions the struct by `cbSize` and accepts every earlier size, so a
    prefix ending at a field both 2.03.06 (the game's DLLs) and 2.03.14 (the
    SDK) know is valid on both. 96 bytes on 64-bit, `randomSeed` at offset 88.
    """

    _fields_ = [
        ("cbSize", ctypes.c_int),
        ("maxMPEGCodecs", ctypes.c_int),
        ("maxADPCMCodecs", ctypes.c_int),
        ("maxXMACodecs", ctypes.c_int),
        ("maxVorbisCodecs", ctypes.c_int),
        ("maxAT9Codecs", ctypes.c_int),
        ("maxFADPCMCodecs", ctypes.c_int),
        ("maxOpusCodecs", ctypes.c_int),
        ("ASIONumChannels", ctypes.c_int),
        ("ASIOChannelList", ctypes.c_void_p),
        ("ASIOSpeakerList", ctypes.c_void_p),
        ("vol0virtualvol", ctypes.c_float),
        ("defaultDecodeBufferSize", ctypes.c_uint),
        ("profilePort", ctypes.c_ushort),
        ("geometryMaxFadeTime", ctypes.c_uint),
        ("distanceFilterCenterFreq", ctypes.c_float),
        ("reverb3Dinstance", ctypes.c_int),
        ("DSPBufferPoolSize", ctypes.c_int),
        ("resamplerMethod", ctypes.c_int),
        ("randomSeed", ctypes.c_uint),
        ("maxConvolutionThreads", ctypes.c_int),
    ]


class FmodStudioAdvancedSettings(ctypes.Structure):
    """`FMOD_STUDIO_ADVANCEDSETTINGS` (fmod_studio_common.h), unchanged
    since 2.00, so the same on the game's 2.03.06 DLLs and the 2.03.14 SDK."""

    _fields_ = [
        ("cbsize", ctypes.c_int),
        ("commandqueuesize", ctypes.c_uint),
        ("handleinitialsize", ctypes.c_uint),
        ("studioupdateperiod", ctypes.c_int),
        ("idlesampledatapoolsize", ctypes.c_int),
        ("streamingscheduledelay", ctypes.c_uint),
        ("encryptionkey", ctypes.c_char_p),
    ]


def _seed(core: ctypes.CDLL, core_system: ctypes.c_void_p) -> None:
    """Fix *core_system*'s random seed. Must run before Initialize."""
    settings = FmodAdvancedSettings(cbSize=ctypes.sizeof(FmodAdvancedSettings))
    _check(core.FMOD_System_GetAdvancedSettings(core_system, ctypes.byref(settings)))
    settings.randomSeed = RANDOM_SEED
    _check(core.FMOD_System_SetAdvancedSettings(core_system, ctypes.byref(settings)))


def _header_version(runtime: FmodRuntime) -> int:
    """FMOD refuses to init if the header version mismatches the library.

    Read it out of the SDK's own header rather than hard-coding a constant that
    silently rots when the SDK is updated. For 2.03.14 this reads 0x00020314.
    The game's DLLs come without headers, so there it is the DLL's own file
    version instead.
    """
    if runtime.header is None:
        return fmod_header_version(*_dll_version(runtime.core))
    m = re.search(r"#define FMOD_VERSION\s+(0x[0-9A-Fa-f]+)", runtime.header.read_text())
    if not m:
        raise SystemExit(f"could not read FMOD_VERSION from {runtime.header}")
    return int(m.group(1), 16)


class EventCensus(NamedTuple):
    """Enumeration tallies, so every dropped entry is accounted for.

    Silent drops are how this module's worst bug hid: three of `Master.bank`'s
    events vanished because `GetPath` failed before the strings bank was
    resident, and nothing said so — the count merely happened to match what
    the plan predicted, for the wrong reason. These counters exist so that
    arithmetic is checked by `_assert_census_balances` rather than by a human
    noticing later.
    """

    reported: int  # summed FMOD_Studio_Bank_GetEventCount across banks
    listed: int  # summed GetEventList 'got' — descriptors actually handed back
    getpath_failed: int  # descriptors whose GetPath did not return FMOD_OK
    snapshots: int  # resolved paths that are snapshot:/ rather than event:/
    duplicates: int  # event paths seen in more than one bank
    paths: list[str]  # unique event:/ paths, sorted


def _assert_census_balances(census: EventCensus) -> None:
    """Fail loudly if the enumeration tallies do not add up.

    `reported` is what the banks claim they hold; every one of those must end
    up in exactly one bucket. If a future FMOD version, bank rebuild, or edit
    to this file starts dropping entries somewhere new, this raises instead of
    silently shipping a smaller sound set.
    """
    if census.listed != census.reported:
        raise RuntimeError(
            f"FMOD bank census mismatch: banks reported {census.reported} events "
            f"but GetEventList handed back {census.listed}"
        )
    accounted = census.getpath_failed + census.snapshots + census.duplicates + len(census.paths)
    if accounted != census.reported:
        raise RuntimeError(
            f"FMOD bank census does not balance: {census.reported} reported != "
            f"{accounted} accounted ({census.getpath_failed} GetPath-failed + "
            f"{census.snapshots} snapshots + {census.duplicates} duplicates + "
            f"{len(census.paths)} unique events)"
        )


def _list_event_paths(
    core: ctypes.CDLL, studio: ctypes.CDLL, version: int, bank_dir: Path
) -> EventCensus:
    """Every event path in the SFX banks, e.g. 'event:/sfx/block_gain'.

    Master.strings.bank is what carries the human-readable paths, which is why
    it is in the allowlist even though it holds no audio.

    Returns an `EventCensus` rather than a bare list so the caller can prove
    nothing was dropped silently — see `_assert_census_balances`.
    """
    system = ctypes.c_void_p()
    _check(studio.FMOD_Studio_System_Create(ctypes.byref(system), version))
    try:
        _check(
            studio.FMOD_Studio_System_Initialize(
                system, 64, FMOD_STUDIO_INIT_NORMAL, FMOD_INIT_NORMAL, None
            )
        )

        paths: list[str] = []
        reported = listed = getpath_failed = snapshots = 0
        # Master.strings.bank MUST load before Master.bank: GetPath is resolved
        # against the strings bank at lookup time, not at load time, so an event
        # queried before the strings bank is resident fails GetPath (counted as
        # getpath_failed below, which makes the census refuse to balance rather
        # than quietly losing the event) — SFX_BANKS itself is left in its
        # Task-1 order (extract_banks doesn't care), so this sorts
        # strings-bank-first just for this loop.
        for rel in sorted(SFX_BANKS, key=lambda b: "strings" not in b):
            bank = ctypes.c_void_p()
            _check(
                studio.FMOD_Studio_System_LoadBankFile(
                    system,
                    str(bank_dir / rel).encode(),
                    FMOD_STUDIO_LOAD_BANK_NORMAL,
                    ctypes.byref(bank),
                )
            )
            count = ctypes.c_int()
            # Checked, not skipped: a bank that cannot report its event count
            # used to `continue` silently, which is indistinguishable from a
            # bank that genuinely holds nothing.
            _check(studio.FMOD_Studio_Bank_GetEventCount(bank, ctypes.byref(count)))
            reported += count.value
            if count.value == 0:
                continue
            arr = (ctypes.c_void_p * count.value)()
            got = ctypes.c_int()
            _check(studio.FMOD_Studio_Bank_GetEventList(bank, arr, count.value, ctypes.byref(got)))
            listed += got.value
            buf = ctypes.create_string_buffer(512)
            for i in range(got.value):
                retrieved = ctypes.c_int()
                if (
                    studio.FMOD_Studio_EventDescription_GetPath(
                        ctypes.c_void_p(arr[i]), buf, 512, ctypes.byref(retrieved)
                    )
                    != FMOD_OK
                ):
                    getpath_failed += 1
                    continue
                path = buf.value.decode()
                # GetEventList also returns snapshots (mixer-state presets,
                # e.g. 'snapshot:/pause' from Master.bank) alongside events.
                # Snapshots carry no audio and event_to_relpath rejects
                # anything without the 'event:/' prefix, so drop them here.
                if path.startswith("event:/"):
                    paths.append(path)
                else:
                    snapshots += 1
    finally:
        # Released on every exit path: without this, a raise anywhere above
        # leaks the native system handle for the life of the process.
        _check(studio.FMOD_Studio_System_Release(system))

    unique = sorted(set(paths))
    return EventCensus(
        reported=reported,
        listed=listed,
        getpath_failed=getpath_failed,
        snapshots=snapshots,
        duplicates=len(paths) - len(unique),
        paths=unique,
    )


# Duration cap for events whose Studio-side envelope never reports STOPPED on
# its own (an AHDSR modulator with a sustain stage needs an explicit stop/
# key-off this batch render never sends, so it holds forever). Measured: with
# a raw 4000-update cap and no duration awareness, 116/567 events rendered to
# a uniform ~85s/16.5MB file instead of a sane length — that is what filled
# the host's disk on the first full run. Capping by rendered SECONDS rather
# than update count keeps that failure mode small and visible instead of
# silent: a look/ambient/sustain event now caps at CAP_SECONDS and gets
# logged, rather than writing an 8x-oversized file with no signal anything
# went wrong.
CAP_SECONDS = 10.0

# Streaming events (`IsStream`, ~130 of 566) are the one place a render can
# depend on wall-clock time. FMOD opens each stream on a thread of its own
# whatever the init flags, and the NRT mixer otherwise races ahead of it: the
# streamed audio then lands a different distance into the render each time,
# and in an event that mixes streamed and preloaded layers the layers drift
# apart (`map_draw` once rendered as pure silence; measured: with no waits all
# seven streaming events probed came out different on every render). Three
# measures, each covering what the others cannot:
#
# 1. STREAM_SCHEDULE_DELAY_SECONDS: Studio starts every streamed instrument
#    this much RENDER time after scheduling it, so the open has that budget
#    (FMOD's default is 8192 samples, ~0.17 s). The delay is leading silence,
#    which encode_sfx trims, and is added to the event's CAP_SECONDS budget.
# 2. STREAM_PACE: a streaming event renders at most this many times faster
#    than real time, so the budget above is ~0.1 s of wall-clock time too.
#    FMOD reports no "opening" state to wait on instead: a stream's channel
#    appears only once it is open, already PLAYING.
# 3. `_wait_for_stream_reads`: a stream REFILL has a state, so before each
#    mixer step the render waits until no sound of the event has a read in
#    flight. Under full CPU load (one busy process per core) a refill once
#    missed its block: two 21 ms blocks of Cubex's charge attack came out
#    different, `diskbusy` set at that very update.
#
# Measured with all three, 4 renders x 13 streaming events: byte-identical
# idle; under full CPU load one stream still opened a 21 ms block late once.
# An open that outlasts the budget (that load, a cold HDD, a virus scanner on
# the bank) shifts a stream, so `_render_stable` renders every event twice
# and keeps it only when the renders agree.
STREAM_SCHEDULE_DELAY_SECONDS = 1.0
STREAM_PACE = 10
# Longest a render waits for one stream read before giving up on the event.
STREAM_READ_TIMEOUT_SECONDS = 10.0


def _event_sounds(core: ctypes.CDLL, group: ctypes.c_void_p, out: list[ctypes.c_void_p]) -> None:
    """Append the current sound of every channel under *group*, recursively."""
    n = ctypes.c_int()
    _check(core.FMOD_ChannelGroup_GetNumChannels(group, ctypes.byref(n)))
    for i in range(n.value):
        channel, sound = ctypes.c_void_p(), ctypes.c_void_p()
        # A channel can end between the count and the lookup: skip it.
        if core.FMOD_ChannelGroup_GetChannel(group, i, ctypes.byref(channel)) != FMOD_OK:
            continue
        if (
            core.FMOD_Channel_GetCurrentSound(channel, ctypes.byref(sound)) == FMOD_OK
            and sound.value
        ):
            out.append(sound)
    n_groups = ctypes.c_int()
    _check(core.FMOD_ChannelGroup_GetNumGroups(group, ctypes.byref(n_groups)))
    for i in range(n_groups.value):
        sub = ctypes.c_void_p()
        _check(core.FMOD_ChannelGroup_GetGroup(group, i, ctypes.byref(sub)))
        _event_sounds(core, sub, out)


def _stream_reading(core: ctypes.CDLL, studio: ctypes.CDLL, inst: ctypes.c_void_p) -> bool:
    """Whether any sound of *inst* has a stream read in flight or is starving."""
    group = ctypes.c_void_p()
    if studio.FMOD_Studio_EventInstance_GetChannelGroup(inst, ctypes.byref(group)) != FMOD_OK:
        return False  # not playing yet (or any more): nothing is streaming
    sounds: list[ctypes.c_void_p] = []
    _event_sounds(core, group, sounds)
    for sound in sounds:
        state, buffered = ctypes.c_int(), ctypes.c_uint()
        starving, disk_busy = ctypes.c_int(), ctypes.c_int()
        result = core.FMOD_Sound_GetOpenState(
            sound,
            ctypes.byref(state),
            ctypes.byref(buffered),
            ctypes.byref(starving),
            ctypes.byref(disk_busy),
        )
        if result == FMOD_OK and (starving.value or disk_busy.value):
            return True
    return False


# Set once a stream read's flag outlasts STREAM_READ_TIMEOUT_SECONDS. The game's
# own FMOD 2.03.06 (`fmod.dll`, used on Windows when FMOD_SDK_ROOT is unset)
# keeps `diskbusy` set between updates for the whole life of a stream: with
# FMOD_INIT_STREAM_FROM_UPDATE the read only happens INSIDE `Update()`, so
# waiting for it before the update never ends. Measured: every streaming event
# timed out and shipped as ~0.1 s of silence (`nibbit_die`, `game_over`, ...),
# while the same events rendered without the wait came out at full length and
# byte-identical across `_render_stable`'s renders. A flag that never clears
# carries no signal, so measure 3 is dropped for the rest of the process.
_stream_wait_disabled = False


def _wait_for_stream_reads(core: ctypes.CDLL, studio: ctypes.CDLL, inst: ctypes.c_void_p) -> None:
    """Block until no stream read of *inst* is in flight (measure 3 above)."""
    global _stream_wait_disabled
    if _stream_wait_disabled:
        return
    deadline = time.monotonic() + STREAM_READ_TIMEOUT_SECONDS
    while _stream_reading(core, studio, inst):
        if time.monotonic() > deadline:
            _stream_wait_disabled = True
            print(
                f"  stream-read flag still set after {STREAM_READ_TIMEOUT_SECONDS:.0f} s: "
                "this FMOD build only reads streams inside Update(); "
                "no longer waiting on stream reads",
                flush=True,
            )
            return
        time.sleep(0.001)


def _apply_render_params(studio: ctypes.CDLL, inst: ctypes.c_void_p, event_path: str) -> None:
    """Set every `RENDER_PARAMS` entry on *inst*, skipping ones it doesn't have.

    `ignoreseekspeed=1` so the value is applied instantly rather than ramped:
    a one-shot render has no time for a seek-speed ramp, and a ramping
    parameter would capture the event mid-transition — a subtler wrong answer
    than not setting it at all.

    `argtypes` is set explicitly, unlike every other call in this file, and
    it is load-bearing: this is the only one taking a FLOAT. Left to ctypes'
    default conversion a Python float would be marshalled as a double and the
    callee would read garbage — silently, since the call still returns
    FMOD_OK. Set once per process; assigning it repeatedly is harmless.
    """
    studio.FMOD_Studio_EventInstance_SetParameterByName.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_float,
        ctypes.c_int,
    ]
    studio.FMOD_Studio_EventInstance_SetParameterByName.restype = ctypes.c_int
    for name, value in RENDER_PARAMS.items():
        result = studio.FMOD_Studio_EventInstance_SetParameterByName(inst, name.encode(), value, 1)
        if result not in (FMOD_OK, FMOD_ERR_EVENT_NOTFOUND):
            raise RuntimeError(f"FMOD error {result} setting {name}={value} on {event_path}")


def _render_one(
    core: ctypes.CDLL,
    studio: ctypes.CDLL,
    version: int,
    bank_dir: Path,
    event_path: str,
    dest: Path,
) -> bool:
    """Render one event to *dest*. Returns True if the render hit CAP_SECONDS
    without the event reporting STOPPED (i.e. it looks like a loop/sustain).

    The WAV-writer driver's output file is fixed at Initialize time, so each
    event needs its own create/init/release cycle. ~570 cycles is slow but
    exact, and this runs once per game version.
    """
    system = ctypes.c_void_p()
    _check(studio.FMOD_Studio_System_Create(ctypes.byref(system), version))
    try:
        core_system = ctypes.c_void_p()
        _check(studio.FMOD_Studio_System_GetCoreSystem(system, ctypes.byref(core_system)))
        _check(core.FMOD_System_SetOutput(core_system, FMOD_OUTPUTTYPE_WAVWRITER_NRT))
        _seed(core, core_system)
        samplerate = ctypes.c_int()
        speakermode = ctypes.c_int()
        numrawspeakers = ctypes.c_int()
        _check(
            core.FMOD_System_GetSoftwareFormat(
                core_system,
                ctypes.byref(samplerate),
                ctypes.byref(speakermode),
                ctypes.byref(numrawspeakers),
            )
        )
        adv = FmodStudioAdvancedSettings(cbsize=ctypes.sizeof(FmodStudioAdvancedSettings))
        _check(studio.FMOD_Studio_System_GetAdvancedSettings(system, ctypes.byref(adv)))
        adv.streamingscheduledelay = round(STREAM_SCHEDULE_DELAY_SECONDS * samplerate.value)
        _check(studio.FMOD_Studio_System_SetAdvancedSettings(system, ctypes.byref(adv)))
        _check(
            studio.FMOD_Studio_System_Initialize(
                system,
                64,
                FMOD_STUDIO_INIT_SYNCHRONOUS_UPDATE | FMOD_STUDIO_INIT_LOAD_FROM_UPDATE,
                FMOD_INIT_STREAM_FROM_UPDATE,
                ctypes.c_char_p(str(dest).encode()),
            )
        )

        # How much rendered audio one Update() advances, read from the system's
        # actual DSP block size and sample rate rather than assumed — the earlier
        # "4000 updates ~= 30s" comment assumed 1024 samples @ 48kHz, which is
        # what let the sustain-envelope bug above go undetected at review time.
        blocklen = ctypes.c_uint()
        numbuffers = ctypes.c_int()
        _check(
            core.FMOD_System_GetDSPBufferSize(
                core_system, ctypes.byref(blocklen), ctypes.byref(numbuffers)
            )
        )
        seconds_per_update = blocklen.value / samplerate.value
        cap_updates = max(1, round(CAP_SECONDS / seconds_per_update))

        for rel in SFX_BANKS:
            bank = ctypes.c_void_p()
            _check(
                studio.FMOD_Studio_System_LoadBankFile(
                    system,
                    str(bank_dir / rel).encode(),
                    FMOD_STUDIO_LOAD_BANK_NORMAL,
                    ctypes.byref(bank),
                )
            )

        desc = ctypes.c_void_p()
        _check(studio.FMOD_Studio_System_GetEvent(system, event_path.encode(), ctypes.byref(desc)))
        # Every take the event could pick is resident before it starts, so the
        # render never waits on a load mid-playback.
        _check(studio.FMOD_Studio_EventDescription_LoadSampleData(desc))
        _check(studio.FMOD_Studio_System_FlushSampleLoading(system))
        inst = ctypes.c_void_p()
        _check(studio.FMOD_Studio_EventDescription_CreateInstance(desc, ctypes.byref(inst)))
        try:
            _apply_render_params(studio, inst, event_path)
            is_stream = ctypes.c_int()
            _check(studio.FMOD_Studio_EventDescription_IsStream(desc, ctypes.byref(is_stream)))
            _check(studio.FMOD_Studio_EventInstance_Start(inst))
            if is_stream.value:
                # The schedule delay is leading silence, not part of the sound.
                cap_updates += round(STREAM_SCHEDULE_DELAY_SECONDS / seconds_per_update)

            state = ctypes.c_int(-1)
            capped = True
            pace = seconds_per_update / STREAM_PACE if is_stream.value else 0.0
            for _ in range(cap_updates):
                if is_stream.value:
                    _wait_for_stream_reads(core, studio, inst)
                _check(studio.FMOD_Studio_System_Update(system))
                if pace:
                    time.sleep(pace)
                _check(studio.FMOD_Studio_EventInstance_GetPlaybackState(inst, ctypes.byref(state)))
                if state.value == FMOD_STUDIO_PLAYBACK_STOPPED:
                    capped = False
                    break
        finally:
            _check(studio.FMOD_Studio_EventInstance_Release(inst))
    finally:
        # Released on every exit path, so one failed event cannot leak a native
        # system handle into the remaining 500-odd renders.
        _check(studio.FMOD_Studio_System_Release(system))
    return capped


# How many times `_render_stable` renders an event, at most, looking for two
# consecutive renders that agree byte for byte.
RENDER_ATTEMPTS = 4


def _render_stable(
    core: ctypes.CDLL,
    studio: ctypes.CDLL,
    version: int,
    bank_dir: Path,
    event_path: str,
    dest: Path,
) -> tuple[bool, bool]:
    """Render *event_path* until two consecutive renders agree; -> (capped, stable).

    The render is meant to be a pure function of the banks, the FMOD build and
    `RANDOM_SEED`, but a streaming event's can still depend on how fast FMOD's
    stream thread runs (see `STREAM_SCHEDULE_DELAY_SECONDS`): measured under
    full CPU load, one stream opened a block late. A second render catches
    that rather than shipping it. Deterministic events agree at once, for the
    price of one extra (fast) render. Not stable after RENDER_ATTEMPTS: the
    last render is kept, and the caller reports it.
    """
    check = dest.with_name(dest.stem + ".check.wav")
    try:
        capped = _render_one(core, studio, version, bank_dir, event_path, dest)
        for _ in range(RENDER_ATTEMPTS - 1):
            capped = _render_one(core, studio, version, bank_dir, event_path, check)
            if check.read_bytes() == dest.read_bytes():
                return capped, True
            check.replace(dest)
        return capped, False
    finally:
        check.unlink(missing_ok=True)


def render_events(bank_dir: Path, out_dir: Path, runtime: FmodRuntime) -> dict[str, Path]:
    """Render every non-music event in the SFX banks to its own WAV.

    Uses FMOD's WAV-writer output driver, so each file is the event as the game
    actually mixes it (layering, randomisation, volume) rather than a raw
    sample. Returns {event path: written file} for the events that succeeded.

    Prints a progress line every 25 events (flushed) so a long run can be told
    "working" from "stuck" — this is ~570 create/init/release cycles and takes
    minutes, not seconds. Also prints every event whose render hit CAP_SECONDS
    without reporting STOPPED (their file length is an artificial cutoff, not
    the sound's real length), every event whose renders never agreed
    (`_render_stable`), and every event whose render FAILED.

    A failing event is logged and skipped rather than aborting the batch: one
    bad event out of ~570 must not cost the other ~569 an hour of rendering.
    Only a run in which *nothing* succeeded raises.
    """
    core, studio, version = _fmod_libs(runtime)
    # No system is created here. `_render_one` creates, initialises and releases
    # its own per event, because the WAV-writer's output file is fixed at
    # Initialize time. A system here too would leak a handle and set the output
    # on a system nothing ever uses.
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    census = _list_event_paths(core, studio, version, bank_dir)
    _assert_census_balances(census)
    event_paths = [p for p in census.paths if not is_music(p)]
    excluded = len(census.paths) - len(event_paths)
    print(
        f"enumerated {census.reported} bank entries: {len(census.paths)} events "
        f"({census.snapshots} snapshots, {census.duplicates} duplicates, "
        f"{census.getpath_failed} unnamed), {excluded} excluded as music, "
        f"{len(event_paths)} to render",
        flush=True,
    )

    total = len(event_paths)
    capped_events: list[str] = []
    failed_events: list[tuple[str, str]] = []
    unstable_events: list[str] = []
    for i, event_path in enumerate(event_paths, start=1):
        dest = out_dir / (event_to_relpath(event_path) + ".wav")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            capped, stable = _render_stable(core, studio, version, bank_dir, event_path, dest)
        except Exception as e:  # noqa: BLE001 — one event must not kill the batch
            failed_events.append((event_path, str(e)))
            print(f"  FAILED {event_path}: {e}", flush=True)
            continue
        if not stable:
            unstable_events.append(event_path)
            print(f"  UNSTABLE (no two renders agreed): {event_path}", flush=True)
        if capped:
            capped_events.append(event_path)
            print(
                f"  CAPPED at {CAP_SECONDS:.0f}s (looks like a loop/sustain): {event_path}",
                flush=True,
            )
        written[event_path] = dest
        if i % 25 == 0 or i == total:
            print(f"{i}/{total} {event_path}", flush=True)

    print(f"{len(capped_events)}/{total} events hit the {CAP_SECONDS:.0f}s cap", flush=True)
    if unstable_events:
        print(
            f"{len(unstable_events)}/{total} events UNSTABLE: rendered {RENDER_ATTEMPTS} times "
            "without two consecutive renders agreeing (the last one is kept)",
            flush=True,
        )
    if failed_events:
        print(f"{len(failed_events)}/{total} events FAILED to render:", flush=True)
        for event_path, err in failed_events:
            print(f"  {event_path}: {err}", flush=True)
    if total and not written:
        raise RuntimeError(f"every one of the {total} events failed to render")
    return written


# Godot's own audio-file extensions inside `res://debug_audio/`. `.tres`
# (a Godot resource — `sleep.tres`) and `.import` (the import sidecar) are
# deliberately absent: neither is a decodable audio file.
DEBUG_AUDIO_SUFFIXES = (".mp3", ".wav", ".ogg")
DEBUG_AUDIO_PREFIX = "debug_audio/"


def extract_debug_audio(pck_path: Path, audio_dir: Path) -> list[Path]:
    """Copy `res://debug_audio/*` out of the pck into `audio_dir/debug_audio/`.

    This is the game's SECOND sound layer and it is not optional. FMOD covers
    most of the game, but `NDebugAudioManager` — whose own doc comment reads
    "Audio manager for use with temporary sound effects (hopefully they make it
    into FMOD someday)" — plays plain Godot streams straight off
    `res://debug_audio/` for a long list of moments the banks have no event
    for at all: combat victory (`NRewardsScreen.cs:400`), obtaining a relic
    (`RelicCmd.cs:49`), the turn banners, card deal/select, potions, and the
    per-card hit layer (`blunt_attack`/`heavy_attack`/`slash_attack`/
    `dagger_throw`, chosen by `AttackCommand.WithHitFx(tmpSfx:)` on 80 cards).
    Extracting only the banks is why those moments were silent in the
    workbench.

    Files are copied VERBATIM, not decoded: `encode_sfx` runs ffmpeg over the
    tree anyway and reads mp3 as happily as wav, so a decode step here would
    only add a lossy generation for nothing.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with open(pck_path, "rb") as f:
        for path, offset, size in _read_pck_directory(f):
            if not path.startswith(DEBUG_AUDIO_PREFIX):
                continue
            if not path.endswith(DEBUG_AUDIO_SUFFIXES):
                continue
            dest = audio_dir / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            f.seek(offset)
            dest.write_bytes(f.read(size))
            written.append(dest)
    if not written:
        raise ValueError(f"pck holds no {DEBUG_AUDIO_PREFIX} audio files")
    return written


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pck", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--render",
        action="store_true",
        help="also decode the extracted banks to named WAVs under out/audio/ "
        "(requires FMOD_SDK_ROOT, or on Windows the game's FMOD DLLs: FMOD_LIB_DIR, "
        "default the pck's directory)",
    )
    args = ap.parse_args()
    runtime = find_fmod_runtime(args.pck.resolve().parent)
    if args.render and runtime is None:
        raise SystemExit(
            "--render needs an FMOD runtime: set FMOD_SDK_ROOT to the unpacked FMOD Studio "
            "API, or on Windows FMOD_LIB_DIR to the game install (it ships fmod.dll "
            "+ fmodstudio.dll)"
        )
    try:
        written = extract_banks(args.pck, args.out)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    for p in written:
        print(f"  wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")

    audio_dir = args.out / "audio"
    try:
        copied = extract_debug_audio(args.pck, audio_dir)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    print(f"  copied {len(copied)} res://debug_audio files to {audio_dir}")

    if args.render:
        assert runtime is not None
        rendered = render_events(args.out, audio_dir, runtime)
        print(f"  rendered {len(rendered)} events to {audio_dir}")


if __name__ == "__main__":
    main()
