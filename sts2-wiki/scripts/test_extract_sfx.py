import ctypes
import struct
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts import extract_sfx
from scripts.extract_sfx import (
    SFX_BANKS,
    EventCensus,
    _assert_census_balances,
    event_to_relpath,
    extract_banks,
    is_music,
    render_events,
)

# A non-allowlisted music bank and an unrelated asset, used to prove the
# allowlist filters rather than merely documents.
MUSIC_BANK = "banks/desktop/act1_a1.bank"
OTHER_ASSET = "localization/eng/cards.json"


def build_pck(entries: list[tuple[str, bytes]], fmt_version: int = 3) -> bytes:
    """Build a minimal in-memory GDPC blob holding *entries* as (path, payload).

    Laid out header → directory → data, with `file_base` pointing at the start
    of the data region and each entry's stored offset relative to it. That is
    what the real pck does, and it means a reader that dropped the
    `file_base + offset` arithmetic would seek into the header and read the
    wrong bytes — so the payload assertions actually cover the seek.
    """
    header_size = 4 + 4 + 12 + 4 + 8 + (8 if fmt_version >= 3 else 0) + 64
    dir_size = 4 + sum(4 + len(p.encode("utf-8")) + 8 + 8 + 16 + 4 for p, _ in entries)
    file_base = header_size + dir_size

    directory = struct.pack("<I", len(entries))
    data = b""
    for path, payload in entries:
        raw = path.encode("utf-8")
        directory += struct.pack("<I", len(raw)) + raw
        directory += struct.pack("<q", len(data))  # offset, relative to file_base
        directory += struct.pack("<q", len(payload))
        directory += b"\x00" * 16  # md5 — the reader skips it
        directory += struct.pack("<I", 0)  # flags
        data += payload

    header = b"GDPC" + struct.pack("<I", fmt_version)
    header += struct.pack("<III", 4, 3, 0)  # engine major/minor/patch
    header += struct.pack("<I", 0)  # pack flags
    header += struct.pack("<Q", file_base)
    if fmt_version >= 3:
        header += struct.pack("<Q", header_size)  # directory offset
    header += b"\x00" * 64  # reserved
    assert len(header) == header_size, (len(header), header_size)
    return header + directory + data


def payload_for(path: str) -> bytes:
    """A distinct, checkable payload per path, of a distinct length."""
    return f"<<{path}>>".encode("utf-8") + b"\xab" * len(path)


class TestBankAllowlist(unittest.TestCase):
    def test_allowlist_is_exactly_the_four_sfx_banks(self):
        self.assertEqual(
            SFX_BANKS,
            (
                "banks/desktop/Master.bank",
                "banks/desktop/Master.strings.bank",
                "banks/desktop/sfx.bank",
                "banks/desktop/temp_sfx.bank",
            ),
        )

    def test_no_music_bank_is_reachable(self):
        # The music banks are ~270 MB and must never be extracted. This asserts
        # the allowlist names none of them, so "no music" cannot regress into a
        # filter someone loosens later.
        for bank in SFX_BANKS:
            self.assertNotIn("act", bank)
            self.assertNotIn("ambience", bank)


class TestExtractBanks(unittest.TestCase):
    """Exercises the pck parser against a synthetic GDPC blob.

    The real pck is gitignored, so these are the only reproducible checks on
    the header layout, the directory walk and the `file_base + offset` seek.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def write_pck(self, paths: list[str], fmt_version: int = 3) -> Path:
        """Write a pck holding *paths* (res:// prefixed) and return its path."""
        entries = [(f"res://{p}", payload_for(p)) for p in paths]
        pck = self.tmp / "test.pck"
        pck.write_bytes(build_pck(entries, fmt_version=fmt_version))
        return pck

    def test_writes_the_allowlisted_banks_with_the_correct_bytes(self):
        # Music first, so a reader that ignored file_base or mis-walked the
        # directory would land on the wrong entry's payload.
        pck = self.write_pck([MUSIC_BANK, *SFX_BANKS, OTHER_ASSET])
        out = self.tmp / "out"

        written = extract_banks(pck, out)

        self.assertEqual(sorted(p.relative_to(out).as_posix() for p in written), sorted(SFX_BANKS))
        for bank in SFX_BANKS:
            self.assertEqual((out / bank).read_bytes(), payload_for(bank), bank)

    def test_does_not_write_non_allowlisted_entries(self):
        pck = self.write_pck([MUSIC_BANK, *SFX_BANKS, OTHER_ASSET])
        out = self.tmp / "out"

        extract_banks(pck, out)

        self.assertFalse((out / MUSIC_BANK).exists())
        self.assertFalse((out / OTHER_ASSET).exists())
        self.assertEqual(
            sorted(p.name for p in (out / "banks/desktop").iterdir()),
            sorted(Path(b).name for b in SFX_BANKS),
        )

    def test_missing_bank_raises_and_names_it(self):
        absent = "banks/desktop/sfx.bank"
        pck = self.write_pck([b for b in SFX_BANKS if b != absent])

        with self.assertRaises(ValueError) as ctx:
            extract_banks(pck, self.tmp / "out")

        self.assertIn(absent, str(ctx.exception))

    def test_missing_bank_raises_a_catchable_exception(self):
        # Regression guard: SystemExit subclasses BaseException, so a caller
        # doing `except Exception` would not catch it. extract_banks is a
        # library interface — only main() may exit the process.
        pck = self.write_pck(list(SFX_BANKS[:2]))

        with self.assertRaises(Exception):
            extract_banks(pck, self.tmp / "out")

    def test_rejects_a_file_that_is_not_a_pck(self):
        pck = self.tmp / "bad.pck"
        pck.write_bytes(b"NOPE" + b"\x00" * 200)

        with self.assertRaises(ValueError) as ctx:
            extract_banks(pck, self.tmp / "out")

        self.assertIn("not a Godot pck", str(ctx.exception))

    def test_reads_the_pre_v3_layout_without_a_directory_offset(self):
        # fmt_version < 3 has no directory-offset field; the directory follows
        # the reserved block inline. Guards the branch in _read_pck_directory.
        pck = self.write_pck(list(SFX_BANKS), fmt_version=2)
        out = self.tmp / "out"

        extract_banks(pck, out)

        for bank in SFX_BANKS:
            self.assertEqual((out / bank).read_bytes(), payload_for(bank), bank)


class TestEventPaths(unittest.TestCase):
    def test_event_becomes_a_relative_file_path(self):
        self.assertEqual(event_to_relpath("event:/sfx/block_gain"), "sfx/block_gain")

    def test_nested_event_keeps_its_directories(self):
        self.assertEqual(
            event_to_relpath(
                "event:/sfx/enemy/enemy_attacks/owl_magistrate/owl_magistrate_attack_peck"
            ),
            "sfx/enemy/enemy_attacks/owl_magistrate/owl_magistrate_attack_peck",
        )

    def test_non_event_string_is_rejected(self):
        with self.assertRaises(ValueError):
            event_to_relpath("sfx/block_gain")


class TestMusicExclusion(unittest.TestCase):
    # Master.bank must be loaded (it carries the mixer) and it ships two
    # menu-music events, so the bank allowlist alone does not satisfy "no music".
    def test_music_namespace_is_excluded(self):
        self.assertTrue(is_music("event:/music/menu/menu_1"))
        self.assertTrue(is_music("event:/music/menu_update"))

    def test_sfx_namespace_is_kept(self):
        self.assertFalse(is_music("event:/sfx/block_gain"))
        self.assertFalse(is_music("event:/temp/sfx/game_over"))

    def test_denylisted_non_music_namespace_event_is_still_excluded(self):
        # event:/temp/temp_music lives in the allowlisted temp_sfx.bank, outside
        # event:/music/, but ffprobe shows 85.7s of audio that never reports
        # STOPPED within the render's ~30s cap (i.e. it loops) — content, not
        # path, says this is music. See SILENTLY_EXCLUDED_EVENTS.
        self.assertTrue(is_music("event:/temp/temp_music"))


def census(**overrides) -> EventCensus:
    """A balancing census, so each test perturbs exactly one field."""
    base = dict(
        reported=4, listed=4, getpath_failed=1, snapshots=1, duplicates=1, paths=["event:/sfx/a"]
    )
    return EventCensus(**{**base, **overrides})


class TestCensusReconciliation(unittest.TestCase):
    """The enumeration must prove nothing was dropped silently.

    Three of Master.bank's events once vanished because GetPath failed before
    the strings bank was resident, and the total still matched what the plan
    predicted — for the wrong reason. These assert the arithmetic that would
    have caught it.
    """

    def test_a_balancing_census_is_accepted(self):
        _assert_census_balances(census())  # 1 + 1 + 1 + 1 == 4

    def test_events_lost_between_count_and_list_are_caught(self):
        with self.assertRaises(RuntimeError) as ctx:
            _assert_census_balances(census(reported=5))

        self.assertIn("reported 5", str(ctx.exception))

    def test_unaccounted_entries_are_caught(self):
        # Banks reported 4 and listed 4, but only 3 ended up in a bucket — the
        # exact shape of the silent-drop bug.
        with self.assertRaises(RuntimeError) as ctx:
            _assert_census_balances(census(duplicates=0))

        self.assertIn("does not balance", str(ctx.exception))


class TestRenderFaultIsolation(unittest.TestCase):
    """One failing event must not cost the other ~569 their render.

    This repo has lost whole label campaigns to a single mid-batch panic, and
    CLAUDE.md requires long jobs to preserve intermediate output. The failure
    is injected rather than provoked with a real render, which takes minutes.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name) / "audio"

        self.events = ["event:/sfx/a", "event:/sfx/boom", "event:/sfx/c"]
        patches = [
            mock.patch.object(
                extract_sfx, "_fmod_libs", return_value=(mock.Mock(), mock.Mock(), 0x00020306)
            ),
            mock.patch.object(
                extract_sfx,
                "_list_event_paths",
                return_value=census(
                    reported=3,
                    listed=3,
                    getpath_failed=0,
                    snapshots=0,
                    duplicates=0,
                    paths=self.events,
                ),
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def render_with(self, render_one):
        with mock.patch.object(extract_sfx, "_render_one", side_effect=render_one):
            return render_events(Path("/nonexistent-banks"), self.out, mock.Mock())

    def test_one_failing_event_does_not_abort_the_batch(self):
        def render_one(core, studio, version, bank_dir, event_path, dest):
            if "boom" in event_path:
                raise RuntimeError("FMOD error 13")
            dest.write_bytes(b"RIFF")
            return False

        written = self.render_with(render_one)

        # The other two still rendered, and the failure is not in the result.
        self.assertEqual(sorted(written), ["event:/sfx/a", "event:/sfx/c"])
        self.assertNotIn("event:/sfx/boom", written)

    def test_a_failure_after_a_success_keeps_the_earlier_work(self):
        # Guards the "lost every event not yet rendered" failure mode: the
        # event that succeeded BEFORE the failure must survive it.
        def render_one(core, studio, version, bank_dir, event_path, dest):
            if event_path != "event:/sfx/a":
                raise RuntimeError("boom")
            dest.write_bytes(b"RIFF")
            return False

        written = self.render_with(render_one)

        self.assertEqual(list(written), ["event:/sfx/a"])

    def test_a_run_where_everything_fails_raises(self):
        def render_one(core, studio, version, bank_dir, event_path, dest):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError) as ctx:
            self.render_with(render_one)

        self.assertIn("every one of the 3 events failed", str(ctx.exception))

    def test_an_unbalanced_census_stops_the_run_before_rendering(self):
        with mock.patch.object(extract_sfx, "_list_event_paths", return_value=census(reported=99)):
            with self.assertRaises(RuntimeError):
                self.render_with(lambda *a: False)

    def test_an_event_whose_renders_never_agree_is_kept_but_reported(self):
        takes = iter(range(100))

        def render_one(core, studio, version, bank_dir, event_path, dest):
            dest.write_bytes(b"RIFF%d" % next(takes))
            return False

        written = self.render_with(render_one)

        # Every event still ships (the last render), and no scratch file stays.
        self.assertEqual(len(written), 3)
        self.assertEqual(
            sorted(p.name for p in self.out.rglob("*.wav")), ["a.wav", "boom.wav", "c.wav"]
        )


class TestRenderStable(unittest.TestCase):
    """A streaming event can render differently when FMOD's stream thread
    falls behind; `_render_stable` re-renders until two consecutive renders
    agree, so a flaky render is caught rather than shipped."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dest = Path(self._tmp.name) / "e.wav"

    def stable_with(self, takes: list[bytes]):
        it = iter(takes)
        calls = []

        def render_one(core, studio, version, bank_dir, event_path, dest):
            calls.append(dest)
            dest.write_bytes(next(it))
            return False

        with mock.patch.object(extract_sfx, "_render_one", side_effect=render_one):
            result = extract_sfx._render_stable(
                mock.Mock(), mock.Mock(), 0, Path("b"), "event:/e", self.dest
            )
        return result, len(calls)

    def test_two_agreeing_renders_are_stable(self):
        (capped, stable), n = self.stable_with([b"x", b"x"])
        self.assertTrue(stable)
        self.assertEqual(n, 2)
        self.assertEqual(self.dest.read_bytes(), b"x")

    def test_a_late_render_is_replaced_by_the_two_that_agree(self):
        (_, stable), n = self.stable_with([b"late", b"x", b"x"])
        self.assertTrue(stable)
        self.assertEqual(n, 3)
        self.assertEqual(self.dest.read_bytes(), b"x")

    def test_never_agreeing_is_unstable_and_keeps_the_last(self):
        takes = [b"%d" % i for i in range(extract_sfx.RENDER_ATTEMPTS)]
        (_, stable), n = self.stable_with(takes)
        self.assertFalse(stable)
        self.assertEqual(n, extract_sfx.RENDER_ATTEMPTS)
        self.assertEqual(self.dest.read_bytes(), takes[-1])
        self.assertEqual([p.name for p in self.dest.parent.iterdir()], ["e.wav"])


class TestApplyRenderParams(unittest.TestCase):
    """The game drives `TakeDamageSfx` with `EnemyImpact_Intensity = 2`
    (`SfxCmd.cs:82`), i.e. on EVERY hit that lands. Rendering at the default
    captured a layer of those events no player has ever heard."""

    def _studio(self, result):
        studio = mock.Mock()
        studio.FMOD_Studio_EventInstance_SetParameterByName = mock.Mock(return_value=result)
        return studio

    def test_sets_every_configured_parameter_on_the_instance(self):
        studio = self._studio(extract_sfx.FMOD_OK)
        inst = object()
        extract_sfx._apply_render_params(studio, inst, "event:/sfx/x")
        call = studio.FMOD_Studio_EventInstance_SetParameterByName.call_args
        self.assertEqual(call.args[0], inst)
        self.assertEqual(call.args[1], b"EnemyImpact_Intensity")
        self.assertEqual(call.args[2], 2.0)
        # ignoreseekspeed=1: a one-shot render has no time for a ramp.
        self.assertEqual(call.args[3], 1)

    def test_declares_the_float_argtype(self):
        # Mutation caught: dropping the argtypes assignment. ctypes would then
        # marshal the value as a double and FMOD would read garbage — while
        # still returning FMOD_OK, so nothing else in this file would notice.
        studio = self._studio(extract_sfx.FMOD_OK)
        extract_sfx._apply_render_params(studio, object(), "event:/sfx/x")
        self.assertEqual(
            studio.FMOD_Studio_EventInstance_SetParameterByName.argtypes[2], ctypes.c_float
        )

    def test_an_event_without_the_parameter_is_left_alone(self):
        # The normal case: 550-odd of the 566 events have no such parameter.
        # Treating that as an error would abort the whole render.
        studio = self._studio(extract_sfx.FMOD_ERR_EVENT_NOTFOUND)
        extract_sfx._apply_render_params(studio, object(), "event:/sfx/x")

    def test_any_other_fmod_error_still_raises(self):
        # Mutation caught: swallowing every non-OK result, which would hide a
        # real failure (a released instance, a bad handle) as "no such param".
        studio = self._studio(31)  # FMOD_ERR_INVALID_PARAM
        with self.assertRaises(RuntimeError) as ctx:
            extract_sfx._apply_render_params(studio, object(), "event:/sfx/x")
        self.assertIn("event:/sfx/x", str(ctx.exception))


class TestStreamFromUpdate(unittest.TestCase):
    """The NRT renderer MUST drive streams from its own update loop.

    Without `FMOD_INIT_STREAM_FROM_UPDATE` the mixer advances hundreds of
    times faster than the stream thread refills its 400 ms buffer, so every
    streaming event renders as that one buffer repeating forever. It hit 135
    of the 566 events — every `_die`, every `_select`, every NPC voice line,
    all three gold cues, `temp/sfx/game_over` — and the symptom downstream was
    either 8 s of stutter or a cue truncated to 0.408 s. There is no cheap
    behavioural test for that (it needs the real SDK, the real banks and an
    ffprobe pass), so this pins the flag itself: the constant must be FMOD's,
    and `_render_one` must actually pass it.
    """

    def test_the_flag_matches_the_sdk_header_value(self):
        self.assertEqual(extract_sfx.FMOD_INIT_STREAM_FROM_UPDATE, 0x00000001)

    def test_render_one_initialises_the_core_system_with_it(self):
        source = Path(extract_sfx.__file__).read_text(encoding="utf-8")
        render_one = source[source.index("def _render_one") : source.index("def render_events")]
        self.assertIn("FMOD_INIT_STREAM_FROM_UPDATE", render_one)
        # And NOT the plain-normal flag it replaced: an `or`-ed pair would
        # still read as "passes the flag" to the check above.
        self.assertNotIn("FMOD_INIT_NORMAL", render_one)


class TestStuckStreamReadFlag(unittest.TestCase):
    """A stream-read flag that never clears must not fail the render.

    The game's FMOD 2.03.06 keeps `diskbusy` set between updates, so waiting
    for it before each `Update()` timed out on every streaming event and the
    import shipped ~0.1 s of silence for each of them.
    """

    def setUp(self):
        patcher = mock.patch.object(extract_sfx, "_stream_wait_disabled", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_flag_that_never_clears_stops_the_waiting_instead_of_raising(self):
        with (
            mock.patch.object(extract_sfx, "STREAM_READ_TIMEOUT_SECONDS", 0.01),
            mock.patch.object(extract_sfx, "_stream_reading", return_value=True) as reading,
        ):
            extract_sfx._wait_for_stream_reads(None, None, None)
            self.assertTrue(extract_sfx._stream_wait_disabled)
            calls = reading.call_count
            extract_sfx._wait_for_stream_reads(None, None, None)
            self.assertEqual(reading.call_count, calls, "a disabled wait must not poll again")

    def test_a_flag_that_clears_keeps_the_wait(self):
        with mock.patch.object(extract_sfx, "_stream_reading", side_effect=[True, False]):
            extract_sfx._wait_for_stream_reads(None, None, None)
        self.assertFalse(extract_sfx._stream_wait_disabled)


class TestExtractDebugAudio(unittest.TestCase):
    """`res://debug_audio/` is the game's second sound layer, not an extra.

    `NDebugAudioManager` plays plain Godot streams for combat victory, relic
    pickups and every card hit sound — moments with no FMOD event at all. The
    banks alone are not the game's sound.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def write_pck(self, paths: list[str]) -> Path:
        entries = [(f"res://{p}", payload_for(p)) for p in paths]
        pck = self.tmp / "test.pck"
        pck.write_bytes(build_pck(entries))
        return pck

    def test_copies_audio_files_verbatim_under_debug_audio(self):
        pck = self.write_pck(["debug_audio/victory.mp3", "debug_audio/ui_click.wav"])
        out = self.tmp / "audio"

        written = extract_sfx.extract_debug_audio(pck, out)

        self.assertEqual(len(written), 2)
        self.assertEqual(
            (out / "debug_audio/victory.mp3").read_bytes(), payload_for("debug_audio/victory.mp3")
        )
        self.assertEqual(
            (out / "debug_audio/ui_click.wav").read_bytes(), payload_for("debug_audio/ui_click.wav")
        )

    def test_skips_the_import_sidecars_and_godot_resources(self):
        # `sleep.tres` is a Godot resource and `*.import` is import metadata;
        # neither decodes, and handing either to ffmpeg would fail the encode
        # run for the whole tree rather than for one file.
        pck = self.write_pck(
            [
                "debug_audio/victory.mp3",
                "debug_audio/victory.mp3.import",
                "debug_audio/sleep.tres",
            ]
        )
        out = self.tmp / "audio"

        written = extract_sfx.extract_debug_audio(pck, out)

        self.assertEqual([p.name for p in written], ["victory.mp3"])

    def test_leaves_everything_outside_debug_audio_alone(self):
        pck = self.write_pck(["debug_audio/victory.mp3", "images/atlases/cards.png"])
        out = self.tmp / "audio"

        extract_sfx.extract_debug_audio(pck, out)

        self.assertFalse((out / "images").exists())

    def test_a_pck_with_no_debug_audio_is_an_error_not_a_silent_skip(self):
        # A game update that moves or renames the directory must be loud: the
        # failure mode it replaces is a workbench that quietly loses victory,
        # relic-get and every card hit sound with nothing in the log.
        pck = self.write_pck(["images/atlases/cards.png"])
        with self.assertRaises(ValueError):
            extract_sfx.extract_debug_audio(pck, self.tmp / "audio")


class TestFmodRuntime(unittest.TestCase):
    """Which FMOD runtime `render_events` loads (`find_fmod_runtime`).

    FMOD_SDK_ROOT keeps winning when it holds this platform's libraries: the
    game's DLLs are a Windows-only fallback, never a replacement.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def files(self, *rels: str) -> Path:
        for rel in rels:
            (self.tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.tmp / rel).write_bytes(b"MZ")
        return self.tmp

    def find(self, env: dict[str, str], os_name: str, default: Path | None = None):
        # The module's own view of `os`: patching the real `os.name` would
        # also turn every Path built during the call into the wrong flavour.
        fake_os = types.SimpleNamespace(name=os_name, environ=env)
        with mock.patch.object(extract_sfx, "os", fake_os):
            return extract_sfx.find_fmod_runtime(default)

    def test_header_version_encodes_each_part_as_decimal_digits(self):
        self.assertEqual(extract_sfx.fmod_header_version(2, 3, 14), 0x00020314)
        self.assertEqual(extract_sfx.fmod_header_version(2, 3, 6), 0x00020306)

    def test_linux_sdk(self):
        sdk = self.files("api/core/lib/x86_64/libfmod.so", "api/studio/lib/x86_64/libfmodstudio.so")
        rt = self.find({"FMOD_SDK_ROOT": str(sdk)}, "posix")
        self.assertIsNotNone(rt)
        self.assertEqual(rt.core, sdk / "api/core/lib/x86_64/libfmod.so")
        self.assertEqual(rt.header, sdk / "api/core/inc/fmod_common.h")

    def test_windows_sdk_uses_its_x64_dlls(self):
        sdk = self.files("api/core/lib/x64/fmod.dll", "api/studio/lib/x64/fmodstudio.dll")
        rt = self.find({"FMOD_SDK_ROOT": str(sdk)}, "nt")
        self.assertIsNotNone(rt)
        self.assertEqual(rt.studio, sdk / "api/studio/lib/x64/fmodstudio.dll")

    def test_sdk_root_wins_over_the_game_dlls(self):
        root = self.files(
            "sdk/api/core/lib/x64/fmod.dll",
            "sdk/api/studio/lib/x64/fmodstudio.dll",
            "game/fmod.dll",
            "game/fmodstudio.dll",
        )
        env = {"FMOD_SDK_ROOT": str(root / "sdk"), "FMOD_LIB_DIR": str(root / "game")}
        self.assertEqual(self.find(env, "nt").core, root / "sdk/api/core/lib/x64/fmod.dll")

    def test_an_sdk_without_this_platforms_libraries_falls_back_to_the_game_dlls(self):
        # A Linux SDK (or a half-unpacked one) on Windows must not crash the
        # render when the game's DLLs are right there.
        root = self.files(
            "sdk/api/core/lib/x86_64/libfmod.so", "game/fmod.dll", "game/fmodstudio.dll"
        )
        rt = self.find({"FMOD_SDK_ROOT": str(root / "sdk")}, "nt", default=root / "game")
        self.assertEqual(
            rt, extract_sfx.FmodRuntime(root / "game/fmod.dll", root / "game/fmodstudio.dll", None)
        )

    def test_fmod_lib_dir_wins_over_the_default(self):
        root = self.files(
            "elsewhere/fmod.dll", "elsewhere/fmodstudio.dll", "pck/fmod.dll", "pck/fmodstudio.dll"
        )
        rt = self.find({"FMOD_LIB_DIR": str(root / "elsewhere")}, "nt", default=root / "pck")
        self.assertEqual(rt.core, root / "elsewhere/fmod.dll")

    def test_game_dlls_are_never_used_off_windows(self):
        root = self.files("fmod.dll", "fmodstudio.dll")
        self.assertIsNone(self.find({"FMOD_LIB_DIR": str(root)}, "posix", default=root))

    def test_windows_lib_dir_needs_both_dlls(self):
        root = self.files("fmod.dll")
        self.assertIsNone(self.find({}, "nt", default=root))
        self.files("fmodstudio.dll")
        self.assertIsNotNone(self.find({}, "nt", default=root))


class TestDeterministicRender(unittest.TestCase):
    """The seed only takes if `FMOD_ADVANCEDSETTINGS` is laid out as FMOD's.

    FMOD validates `cbSize` against its known struct versions, so a wrong size
    fails loudly; a wrong field ORDER would not, and would set some other field
    instead of the seed. These are the offsets fmod_common.h gives on 64-bit.
    """

    @unittest.skipUnless(ctypes.sizeof(ctypes.c_void_p) == 8, "64-bit layout")
    def test_advanced_settings_layout_matches_fmod_common_h(self):
        s = extract_sfx.FmodAdvancedSettings
        self.assertEqual(ctypes.sizeof(s), 96)
        self.assertEqual(s.ASIOChannelList.offset, 40)
        self.assertEqual(s.defaultDecodeBufferSize.offset, 60)
        self.assertEqual(s.randomSeed.offset, 88)

    def test_seed_is_nonzero(self):
        # 0 is FMOD's "seed from the clock".
        self.assertNotEqual(extract_sfx.RANDOM_SEED, 0)
