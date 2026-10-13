import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import encode_sfx
from scripts.encode_sfx import (
    LOOP_SR,
    PEAK_LIMIT_DBFS,
    cap_for,
    corpus_gain_db,
    detect_loop,
    encode_tree,
    ffmpeg_args,
    measure_peak_dbfs,
    parse_peak_dbfs,
    trim_window,
    verify_peaks,
    window_peak_args,
)

# A real `astats` stderr excerpt (mono, `metadata=0`) — one Overall section,
# as ffmpeg actually emits it for the files this script produces.
ASTATS_OUTPUT = """
[Parsed_astats_0 @ 0x1] Overall
[Parsed_astats_0 @ 0x1] DC offset: 0.000195
[Parsed_astats_0 @ 0x1] Min level: -1.228715
[Parsed_astats_0 @ 0x1] Max level: 1.037562
[Parsed_astats_0 @ 0x1] Peak level dB: 1.789020
[Parsed_astats_0 @ 0x1] RMS level dB: -21.030006
"""

# Real `silencedetect` output shapes. ffmpeg emits these on stderr, one pair of
# lines per silent run.
LEADING_THEN_GAP = """
[silencedetect @ 0x1] silence_start: 0
[silencedetect @ 0x1] silence_end: 1.02 | silence_duration: 1.02
[silencedetect @ 0x1] silence_start: 3.51
[silencedetect @ 0x1] silence_end: 5.2 | silence_duration: 1.69
[silencedetect @ 0x1] silence_start: 8.4
"""

NO_SILENCE_AT_ALL = ""

STARTS_LOUD_THEN_ENDS = """
[silencedetect @ 0x1] silence_start: 2.25
[silencedetect @ 0x1] silence_end: 4.0 | silence_duration: 1.75
"""


class TestTrimWindow(unittest.TestCase):
    def test_skips_leading_silence_and_cuts_at_the_first_gap(self):
        # One greeting out of a looped event: start where audio starts, end
        # where it first falls silent again.
        self.assertEqual(trim_window(LEADING_THEN_GAP, 85.76), (1.02, 2.49))

    def test_file_that_starts_loud_starts_at_zero(self):
        self.assertEqual(trim_window(STARTS_LOUD_THEN_ENDS, 30.0), (0.0, 2.25))

    def test_no_silence_falls_back_to_the_hard_cap(self):
        # A genuine ambience loop never goes quiet; bound it rather than ship
        # 85 seconds of it.
        self.assertEqual(trim_window(NO_SILENCE_AT_ALL, 85.76), (0.0, 8.0))

    def test_short_file_with_no_silence_keeps_its_full_length(self):
        self.assertEqual(trim_window(NO_SILENCE_AT_ALL, 0.4), (0.0, 0.4))

    def test_window_never_collapses_to_nothing(self):
        # Defensive: a file detected as silent end-to-end must not encode to a
        # zero-length cue, which would be an unplayable file rather than a
        # quiet one.
        allsilent = "silence_start: 0\nsilence_end: 9.0 | silence_duration: 9.0\n"
        start, length = trim_window(allsilent, 9.0)
        self.assertGreater(length, 0.0)


class TestCapFor(unittest.TestCase):
    def test_defaults_to_the_global_cap(self):
        self.assertEqual(cap_for("sfx/block_gain.wav"), 8.0)

    def test_longest_matching_prefix_wins(self):
        # Guards the retune seam: a narrower prefix must beat a wider one rather
        # than the result depending on dict insertion order.
        from scripts import encode_sfx

        original = dict(encode_sfx.CAP_BY_PREFIX)
        encode_sfx.CAP_BY_PREFIX.update({"sfx/ui/": 2.0, "sfx/ui/gold/": 1.2})
        try:
            self.assertEqual(encode_sfx.cap_for("sfx/ui/map/map_open.wav"), 2.0)
            self.assertEqual(encode_sfx.cap_for("sfx/ui/gold/gold_1.wav"), 1.2)
            self.assertEqual(encode_sfx.cap_for("sfx/npcs/neow/x.wav"), 8.0)
        finally:
            encode_sfx.CAP_BY_PREFIX.clear()
            encode_sfx.CAP_BY_PREFIX.update(original)


def _autocorr(samples: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation, mirroring `detect_loop`'s internal formula.

    Duplicated here (not imported) so the adversarial test below can assert
    on the intermediate correlation values without depending on `detect_loop`
    exposing internals it otherwise has no reason to expose.
    """
    x = samples.astype(np.float64)
    x = x - x.mean()
    n = len(x)
    spectrum = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    result: np.ndarray = ac / ac[0]
    return result


class TestDetectLoop(unittest.TestCase):
    def test_a_tiled_block_is_detected_at_its_own_period(self):
        rng = np.random.default_rng(42)
        period_s = 0.4
        block = rng.standard_normal(int(period_s * LOOP_SR)).astype(np.float32)
        samples = np.tile(block, 10)
        period = detect_loop(samples, LOOP_SR)
        self.assertIsNotNone(period)
        # Within a few ms — autocorrelation resolves to the nearest sample.
        self.assertAlmostEqual(period, period_s, delta=0.005)

    def test_a_single_decaying_transient_is_not_a_loop(self):
        rng = np.random.default_rng(1)
        n = int(2.0 * LOOP_SR)
        envelope = np.exp(-np.arange(n) / (0.05 * LOOP_SR))
        samples = (rng.standard_normal(n) * envelope).astype(np.float32)
        self.assertIsNone(detect_loop(samples, LOOP_SR))

    def test_white_noise_is_not_a_loop(self):
        rng = np.random.default_rng(2)
        samples = rng.standard_normal(int(2.0 * LOOP_SR)).astype(np.float32)
        self.assertIsNone(detect_loop(samples, LOOP_SR))

    def test_a_file_too_short_to_hold_four_periods_returns_none(self):
        rng = np.random.default_rng(3)
        # Under a second, well short of the 4x0.15s = 0.6s floor needed to
        # even start the lag search, let alone hold 4 repetitions of anything.
        samples = rng.standard_normal(int(0.3 * LOOP_SR)).astype(np.float32)
        self.assertIsNone(detect_loop(samples, LOOP_SR))  # must not raise

    def test_strong_fundamental_with_weak_harmonics_is_rejected(self):
        # Regression for the exact mistake the 3-condition rule exists to
        # prevent: judging on the autocorrelation fundamental alone got this
        # wrong twice during design (once false-accepting a voice line, once
        # false-rejecting a real loop after over-tightening). A block tiled
        # exactly 4x gives an EXACT, seed-independent autocorrelation ratio
        # of (N-k)/N at lag k*period — for N=4 that's fundamental=0.75 (just
        # clears LOOP_FUND_THRESHOLD) but 2nd harmonic=0.50 (fails
        # LOOP_HARMONIC2_THRESHOLD=0.55) and 3rd=0.25 (fails
        # LOOP_HARMONIC3_THRESHOLD=0.40). If someone simplified the accept
        # rule to "ac[lag] >= LOOP_FUND_THRESHOLD" alone, this exact case
        # would flip from rejected to a false-positive 0.4s loop — every one
        # of the other tests in this class would still pass, which is what
        # makes this the one worth pinning explicitly.
        rng = np.random.default_rng(11)
        period_s = 0.4
        block = rng.standard_normal(int(period_s * LOOP_SR)).astype(np.float32)
        samples = np.tile(block, 4)

        # Assert the adversarial shape itself, not just the outcome: a real
        # reader should be able to see *why* this is the case that catches a
        # single-condition regression.
        ac = _autocorr(samples)
        lag = int(period_s * LOOP_SR)
        self.assertGreaterEqual(ac[lag], 0.75)  # fundamental alone would accept
        self.assertLess(ac[2 * lag], 0.55)  # ...but the 2nd harmonic is too weak
        self.assertLess(ac[3 * lag], 0.40)  # ...and so is the 3rd

        self.assertIsNone(detect_loop(samples, LOOP_SR))


class TestFfmpegArgs(unittest.TestCase):
    def test_encodes_mono_opus_at_64k(self):
        args = ffmpeg_args(Path("a.wav"), Path("a.webm"), 0.0, 1.0, -3.0)
        self.assertIn("libopus", args)
        self.assertIn("-b:a", args)
        self.assertIn("64k", args)
        self.assertIn("-ac", args)
        self.assertIn("1", args)

    def test_applies_the_corpus_gain_and_never_per_file_loudness(self):
        # `loudnorm` is what flattened the mix: a UI click and a boss slam
        # both came out at -16 LUFS, so the workbench could not sound like
        # the game no matter how correct the event mapping was. Asserting its
        # ABSENCE is the point of this test — the positive `volume=` check
        # alone would still pass if both filters were applied.
        joined = " ".join(ffmpeg_args(Path("a.wav"), Path("a.webm"), 0.0, 1.0, -7.25))
        self.assertNotIn("loudnorm", joined)
        self.assertIn("volume=-7.2500dB", joined)

    def test_the_gain_is_whatever_the_caller_computed(self):
        # Mutation caught: hard-coding TARGET_TP (or any constant) in place of
        # the argument, which would put every corpus at the same level again.
        self.assertIn(
            "volume=2.5000dB", " ".join(ffmpeg_args(Path("a.wav"), Path("a.webm"), 0.0, 1.0, 2.5))
        )

    def test_applies_the_trim_window(self):
        args = ffmpeg_args(Path("a.wav"), Path("a.webm"), 1.02, 2.49, -3.0)
        # -ss must precede -i so ffmpeg seeks rather than decoding and dropping.
        self.assertLess(args.index("-ss"), args.index("-i"))
        self.assertIn("1.02", args)
        self.assertIn("2.49", args)

    def test_overwrites_without_prompting(self):
        self.assertIn("-y", ffmpeg_args(Path("a.wav"), Path("a.webm"), 0.0, 1.0, -3.0))

    def test_output_is_reproducible(self):
        # Container and codec both bitexact: no random UIDs or version tags.
        joined = " ".join(ffmpeg_args(Path("a.wav"), Path("a.webm"), 0.0, 1.0, -3.0))
        self.assertIn("-fflags +bitexact", joined)
        self.assertIn("-flags:a +bitexact", joined)


class TestCorpusGainDb(unittest.TestCase):
    def test_puts_the_loudest_window_at_the_target(self):
        self.assertAlmostEqual(corpus_gain_db([-20.0, -2.0, -40.0], target_tp=-6.0), -4.0)

    def test_preserves_the_distance_between_files(self):
        # The whole point. Two files 18 dB apart must still be 18 dB apart
        # after the gain — mutation caught: returning a per-file gain, or
        # clamping quiet files up toward the target.
        peaks = [-2.0, -20.0]
        gain = corpus_gain_db(peaks, target_tp=-6.0)
        after = [p + gain for p in peaks]
        self.assertAlmostEqual(after[0] - after[1], peaks[0] - peaks[1])
        self.assertAlmostEqual(max(after), -6.0)

    def test_ignores_digitally_silent_files(self):
        # A handful of events render to pure silence; astats reports -inf.
        # Letting one into the max would make the gain +inf.
        gain = corpus_gain_db([float("-inf"), -10.0], target_tp=-6.0)
        self.assertAlmostEqual(gain, 4.0)

    def test_an_all_silent_corpus_is_a_no_op_not_an_error(self):
        self.assertEqual(corpus_gain_db([float("-inf")], target_tp=-6.0), 0.0)
        self.assertEqual(corpus_gain_db([], target_tp=-6.0), 0.0)


class TestWindowPeakArgs(unittest.TestCase):
    def test_measures_the_trimmed_window_not_the_whole_file(self):
        # Mutation caught: measuring the untrimmed file, which would let a
        # loud lead-in or a spike in a cut tail set the gain for all 566.
        args = window_peak_args(Path("a.wav"), 1.02, 2.49)
        self.assertLess(args.index("-ss"), args.index("-i"))
        self.assertIn("1.02", args)
        self.assertIn("2.49", args)
        self.assertIn("astats=metadata=0", args)


class TestParsePeakDbfs(unittest.TestCase):
    def test_reads_the_overall_peak_level(self):
        self.assertAlmostEqual(parse_peak_dbfs(ASTATS_OUTPUT), 1.789020)

    def test_raises_when_astats_carried_no_reading(self):
        # A parse failure must be loud, not a silent "0 dB" default that would
        # let a real clip through undetected.
        with self.assertRaises(ValueError):
            parse_peak_dbfs("no astats lines here")

    def test_reads_negative_infinity_for_pure_silence(self):
        # A handful of the real 566 render as pure digital silence (a Task 2
        # artifact) and ffmpeg reports that as the literal `-inf`, not a
        # number — this must parse, not crash the whole run on one file.
        silent = "[Parsed_astats_0 @ 0x1] Peak level dB: -inf\n"
        self.assertEqual(parse_peak_dbfs(silent), float("-inf"))


def _synth_webm(dest: Path, volume_db: float) -> None:
    """Encode a short full-scale sine to *dest*, boosted by *volume_db*.

    Bypasses this script's own encode pipeline (loudnorm/TARGET_TP) on
    purpose: these tests need to hand `measure_peak_dbfs`/`verify_peaks` a
    file whose *decoded* loudness they control directly, to prove the
    post-encode check catches a real clip rather than trusting the encoder
    settings that produced it.
    """
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=0.3",
            "-af",
            f"volume={volume_db}dB",
            "-c:a",
            "libopus",
            "-b:a",
            "64k",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )


class TestVerifyPeaks(unittest.TestCase):
    """Integration coverage: real ffmpeg-encoded files, not mocked output.

    A synthetic *quiet* file (attenuated sine) must pass; a synthetic *loud*
    one (a full-scale sine boosted well past 0 dBFS, so Opus reconstructs it
    over the limit on decode) must be caught and named. This is the case the
    shipped bug fell through: nothing measured the actual decoded artifact.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.quiet = Path(self.tmp.name) / "quiet.webm"
        self.loud = Path(self.tmp.name) / "loud.webm"
        # lavfi's `sine` source is well below full scale (0 dB gain measures
        # ~-18 dBFS here), so the "loud" boost needs to be large, not just
        # positive, to land clearly over PEAK_LIMIT_DBFS.
        _synth_webm(self.quiet, volume_db=-20)
        _synth_webm(self.loud, volume_db=22)

    def test_measures_a_quiet_file_under_the_limit(self):
        self.assertLess(measure_peak_dbfs(self.quiet), PEAK_LIMIT_DBFS)

    def test_measures_a_loud_file_over_the_limit(self):
        self.assertGreater(measure_peak_dbfs(self.loud), PEAK_LIMIT_DBFS)

    def test_passes_when_every_file_is_within_the_limit(self):
        verify_peaks([self.quiet])  # must not raise

    def test_fails_loudly_and_names_the_offending_file(self):
        with self.assertRaises(RuntimeError) as ctx:
            verify_peaks([self.quiet, self.loud])
        self.assertIn(str(self.loud), str(ctx.exception))
        self.assertNotIn(str(self.quiet), str(ctx.exception))


def _write_short_wav(dest: Path, duration: float = 0.4) -> None:
    """Write a short synthetic tone at *dest*, creating parent dirs as needed.

    Real ffmpeg-encoded audio, not a hand-rolled WAV header, so it exercises
    the same `silencedetect`/`ffprobe`/decode passes `encode_tree` runs on
    real extracted cues.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )


class TestEncodeTree(unittest.TestCase):
    def test_produces_one_playable_webm_per_input(self):
        with (
            tempfile.TemporaryDirectory() as src_dir,
            tempfile.TemporaryDirectory() as dest_dir,
        ):
            src, dest = Path(src_dir), Path(dest_dir)
            _write_short_wav(src / "sfx" / "a.wav")
            _write_short_wav(src / "sfx" / "ui" / "b.wav")

            written = encode_tree(src, dest)

            self.assertEqual(len(written), 2)
            for path in written:
                self.assertTrue(path.exists())
                self.assertEqual(path.suffix, ".webm")
                # Not just "a .webm exists at dest" but specifically the
                # renamed-into-place file, not the hidden per-process temp
                # name (`.<stem>.<pid>.tmp.webm`) left in place by a broken
                # rename step.
                self.assertFalse(path.name.startswith("."))
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", str(path)],
                    capture_output=True,
                )
                self.assertEqual(probe.returncode, 0, probe.stderr)
            # The rename step must leave nothing behind under its own
            # (hidden, dotted) temp name once every file succeeds.
            self.assertEqual(list(dest.rglob("*.tmp.webm")), [])

    def test_failed_encode_leaves_no_tmp_file_behind(self):
        # Regression guard: an encode that dies mid-write (ffmpeg crash, OOM
        # kill, full disk) must not leave a `.tmp.webm` in the shipped tree.
        # Reproduced concretely once by kill -9'ing a real ffmpeg encode
        # mid-stream, which left a partial file whose ffprobe failed with
        # "EBML header parsing failed" — the same signature the atomic
        # os.replace was written to prevent, just one path segment over.
        with (
            tempfile.TemporaryDirectory() as src_dir,
            tempfile.TemporaryDirectory() as dest_dir,
        ):
            src, dest = Path(src_dir), Path(dest_dir)
            _write_short_wav(src / "sfx" / "a.wav")

            real_run = subprocess.run

            def flaky_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
                # Only the final Opus encode carries "libopus" — every other
                # subprocess.run call in encode_tree (silencedetect, ffprobe
                # duration, the loop-detection decode, astats) must still run
                # for real so the failure is isolated to the encode step.
                if "libopus" in cmd:
                    # A killed/crashed ffmpeg leaves a partial file on disk —
                    # the mock must reproduce that (not just raise) or the
                    # "no leftover .tmp.webm" assertion below is vacuous: it
                    # would pass even with tmp_dest.unlink() deleted, since
                    # no file would ever exist either way. cmd's last element
                    # is ffmpeg_args' output path, i.e. tmp_dest itself.
                    Path(cmd[-1]).write_bytes(b"\x1a\x45\xdf\xa3partial")
                    raise subprocess.CalledProcessError(1, cmd)
                return real_run(cmd, *args, **kwargs)

            with mock.patch("scripts.encode_sfx.subprocess.run", side_effect=flaky_run):
                with self.assertRaises(subprocess.CalledProcessError):
                    encode_tree(src, dest)

            leftover = list(dest.rglob("*.tmp.webm"))
            self.assertEqual(leftover, [])

    def test_keyboard_interrupt_during_encode_leaves_no_tmp_file_behind(self):
        # KeyboardInterrupt/SystemExit derive from BaseException, not
        # Exception — an `except Exception` cleanup block silently skips
        # them. Ctrl-C mid-run is the single most likely way this long,
        # human-tailed job gets interrupted, so this must be covered
        # explicitly, not just inferred from the CalledProcessError case
        # above. Fails against a bare `except Exception:`, passes with the
        # `finally` + success-flag form.
        with (
            tempfile.TemporaryDirectory() as src_dir,
            tempfile.TemporaryDirectory() as dest_dir,
        ):
            src, dest = Path(src_dir), Path(dest_dir)
            _write_short_wav(src / "sfx" / "a.wav")

            real_run = subprocess.run

            def interrupted_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
                if "libopus" in cmd:
                    # Same reasoning as flaky_run above: a real Ctrl-C lands
                    # mid-write, so the mock must leave a partial file behind
                    # (cmd[-1] is tmp_dest) or "no leftover .tmp.webm" is
                    # trivially true regardless of whether cleanup runs.
                    Path(cmd[-1]).write_bytes(b"\x1a\x45\xdf\xa3partial")
                    raise KeyboardInterrupt
                return real_run(cmd, *args, **kwargs)

            with mock.patch("scripts.encode_sfx.subprocess.run", side_effect=interrupted_run):
                with self.assertRaises(KeyboardInterrupt):
                    encode_tree(src, dest)

            leftover = list(dest.rglob("*.tmp.webm"))
            self.assertEqual(leftover, [])


if __name__ == "__main__":
    unittest.main()


class TestSourceFiles(unittest.TestCase):
    """Both source families go through ONE encode pass.

    The FMOD renderer writes `.wav`; `extract_sfx` copies the game's
    `res://debug_audio/` layer out as `.mp3`. They are mixed against each
    other in the game, so `corpus_gain_db` has to see both — measuring them
    in separate passes would put the two layers at the wrong relative level,
    which is the same mistake per-file loudnorm made.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _touch(self, rel: str) -> None:
        path = self.tmp / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00")

    def test_walks_wav_and_mp3_together(self):
        self._touch("sfx/block_gain.wav")
        self._touch("debug_audio/victory.mp3")

        found = [p.relative_to(self.tmp).as_posix() for p in encode_sfx.source_files(self.tmp)]

        self.assertEqual(found, ["debug_audio/victory.mp3", "sfx/block_gain.wav"])

    def test_ignores_everything_that_is_not_a_cue(self):
        # The pck's import sidecars are filtered at extraction, but a stray
        # `.json`/`.tres` in the tree must not reach ffmpeg either.
        self._touch("sfx/block_gain.wav")
        self._touch("debug_audio/sleep.tres")
        self._touch("notes.json")

        found = [p.name for p in encode_sfx.source_files(self.tmp)]

        self.assertEqual(found, ["block_gain.wav"])

    def test_is_case_insensitive_on_the_extension(self):
        self._touch("debug_audio/VICTORY.MP3")
        self.assertEqual([p.name for p in encode_sfx.source_files(self.tmp)], ["VICTORY.MP3"])
