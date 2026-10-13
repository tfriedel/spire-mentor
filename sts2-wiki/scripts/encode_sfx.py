#!/usr/bin/env python3
"""Transcode extracted SFX WAVs to Opus-in-WebM, preserving the game's mix.

Mono, 64 kbps. Every file gets ONE shared gain, not per-file normalisation —
see `corpus_gain_db`. Chrome and Firefox on Linux are the only targets, so
there are no codec fallbacks.

The constraint that actually matters is on the shipped artifact: every
`.webm` must measure at or under `PEAK_LIMIT_DBFS` on playback. `loudnorm`'s
`TP` target only bounds the pre-encode PCM — Opus at 64 kbps can reconstruct
peaks above it on decode — so `encode_tree` measures each finished `.webm`
with `verify_peaks` and fails loudly, naming files, rather than trusting the
encoder setting. See `TARGET_TP` for how that setting was chosen.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np

SILENCE_DB = -50
SILENCE_MIN = 0.35
HARD_CAP = 8.0

# Loop-detection decode/search parameters. Measured on real files via
# normalised FFT autocorrelation of the post-lead-in audio: a real loop (e.g.
# `gold_1`, period 0.400 s) has a strong harmonic series at multiples of its
# period (+0.89/+0.77/+0.66 at 1x/2x/3x lag); a one-shot (`block_gain`) scores
# near zero even at its best lag (+0.03). The gap between the two classes is
# enormous, so these thresholds are not delicate — but all three harmonics are
# required, not just the fundamental: a lone peak at the fundamental alone is
# not enough to separate a real loop from a spurious narrow correlation peak.
LOOP_SR = 16000
LOOP_MIN_LAG_S = 0.15
LOOP_MAX_LAG_S = 4.0
LOOP_FUND_THRESHOLD = 0.75
LOOP_HARMONIC2_THRESHOLD = 0.55
LOOP_HARMONIC3_THRESHOLD = 0.40

# Where the LOUDEST window in the corpus lands, pre-encode. Every other file
# sits below it by whatever margin the game's own mix gives it -- this is the
# single anchor `corpus_gain_db` solves for, not a per-file target.
#
# NOT the shipped constraint — that's PEAK_LIMIT_DBFS below, checked
# post-encode. This is headroom against Opus's post-decode overshoot, and the
# measurements behind the number were taken when this was loudnorm's `TP=`
# (which put EVERY file at the ceiling, so they bound the worst case far more
# tightly than the constant-gain scheme needs): at TP=-1.5 a 40-file sample showed
# 70-78% of outputs decoding above -1.5 dBTP and 35-52% above 0 dBFS — real
# clipping. A first pass at TP=-4.5, tuned against a 3-file sample, still left
# 9 of the real 566 over PEAK_LIMIT_DBFS (worst: -0.64 dBFS, 0.36 dB over) —
# a 3-file sample doesn't cover the population's tail. -6.0 (also measured
# safe on that same 3-file sample: -2.74/-4.65/-2.95 dBFS) verified clean
# across all 566 — see verify_peaks, which is the actual enforcement and
# would catch a future regression here regardless of this constant.
TARGET_TP = -6.0

# The shipped .webm must measure at or under this many dBFS. Enforced by
# verify_peaks against the actual decoded output, not assumed from TARGET_TP.
PEAK_LIMIT_DBFS = -1.0

# Print a progress line every N files so a long encode run shows "working" vs
# "stuck" (per this repo's CLAUDE.md rule on long-running jobs).
PROGRESS_EVERY = 25

# The cap is per-namespace, not global, because for a large minority of events
# the cap IS the length. 75 of the 566 never fall silent at any threshold — the
# game drives them and calls Stop() itself — so the silence detector never fires
# and the cap alone decides. A coin pickup and an NPC voice line want very
# different numbers, and tightening the silence threshold instead is measurably
# worse: it damages short cues (ui_click 0.27 s -> 0.04 s, cutting the click's
# tail) while doing nothing at all for the long ones.
#
# Longest matching prefix wins; "" is the default. Retuning is deliberately a
# data edit here, so a listening pass costs one line and a re-encode.
CAP_BY_PREFIX: dict[str, float] = {
    "": 8.0,
}


def cap_for(rel: str) -> float:
    """Cap in seconds for the file at *rel* (e.g. `sfx/ui/gold/gold_1.wav`)."""
    best, cap = -1, HARD_CAP
    for prefix, value in CAP_BY_PREFIX.items():
        if rel.startswith(prefix) and len(prefix) > best:
            best, cap = len(prefix), value
    return cap


def detect_args(src: Path) -> list[str]:
    """ffmpeg invocation whose stderr carries the silence map for *src*."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(src),
        "-af",
        f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN}",
        "-f",
        "null",
        "-",
    ]


def trim_window(
    detect_output: str, duration: float, hard_cap: float = HARD_CAP
) -> tuple[float, float]:
    """Pick the (start, length) of the cue's first coherent burst.

    Start at the first sample of actual audio, so a cue fires the instant it is
    triggered rather than after a second of leading silence. End at the first
    time it falls silent again, which for a looped event is the end of the first
    repetition — one greeting, not eighty seconds of them.

    Pure so it is testable: it parses `silencedetect`'s stderr rather than
    running ffmpeg.
    """
    starts: list[float] = []
    ends: list[float] = []
    for line in detect_output.splitlines():
        if "silence_start:" in line:
            starts.append(float(line.split("silence_start:")[1].split("|")[0].strip()))
        elif "silence_end:" in line:
            ends.append(float(line.split("silence_end:")[1].split("|")[0].strip()))

    # Leading silence: a run starting at (or within a hair of) zero.
    begin = ends[0] if starts and starts[0] <= 0.01 and ends else 0.0
    # The first time it goes quiet again after that.
    after = [t for t in starts if t > begin + 0.01]
    end = after[0] if after else min(begin + hard_cap, duration)
    length = min(end - begin, hard_cap)
    # Never emit a zero-length cue: an unplayable file is worse than a quiet one.
    if length <= 0.0:
        begin, length = 0.0, min(duration, hard_cap)
    return round(begin, 3), round(length, 3)


def detect_loop(samples: np.ndarray, sr: int) -> float | None:
    """Detect a strong repeating period in *samples*, or None if it isn't a loop.

    *samples* is mono PCM already past any lead-in — the loop-detection window
    is the cue's body, not the whole file. Pure/testable: takes decoded
    samples directly rather than running ffmpeg itself (mirrors the
    `trim_window` / `detect_args` split).

    Method: normalised FFT autocorrelation, searched over lags from
    `LOOP_MIN_LAG_S` to `min(LOOP_MAX_LAG_S, n/4 samples)` — capping at n/4
    guarantees enough signal remains to check the 3rd harmonic at the winning
    lag, so a file too short to hold ~4 repetitions returns None outright
    rather than measuring noise. A candidate lag is accepted as the loop
    period only if it and its 2nd and 3rd harmonics all clear their
    thresholds (see the module-level comment for why the fundamental alone
    isn't sufficient).
    """
    n = samples.shape[0]
    min_lag = int(round(LOOP_MIN_LAG_S * sr))
    max_lag = int(min(LOOP_MAX_LAG_S * sr, n / 4))
    if max_lag <= min_lag:
        return None

    x = samples.astype(np.float64)
    x = x - x.mean()
    nfft = 2 * n
    spectrum = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]

    window = ac[min_lag : max_lag + 1]
    if window.size == 0:
        return None
    lag = min_lag + int(np.argmax(window))

    def ac_at(k: int) -> float:
        return float(ac[k]) if k < n else -1.0

    if (
        ac_at(lag) >= LOOP_FUND_THRESHOLD
        and ac_at(2 * lag) >= LOOP_HARMONIC2_THRESHOLD
        and ac_at(3 * lag) >= LOOP_HARMONIC3_THRESHOLD
    ):
        return lag / sr
    return None


def _decode_samples(src: Path, start: float, sr: int = LOOP_SR) -> np.ndarray:
    """Decode *src* to mono float32 PCM at *sr* Hz, seeking past *start* first."""
    out = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start),
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(out.stdout, dtype=np.float32)


def ffmpeg_args(src: Path, dest: Path, start: float, length: float, gain_db: float) -> list[str]:
    """The encode invocation for one trimmed window at the corpus-wide gain.

    `volume`, not `loudnorm`. This used to be `loudnorm=I=-16:LRA=11`, which
    dragged every cue to the same perceived loudness — a UI click and a boss
    slam came out equally loud, which is the single biggest reason the
    workbench did not sound like the game even with every event mapped
    correctly. FMOD already renders each event at its authored volume, so the
    WAVs carry the game's own relative levels; the only thing needed is one
    shared gain that fits the loudest of them under the ceiling. See
    `corpus_gain_db`.
    """
    return [
        "ffmpeg",
        "-y",
        # -ss before -i so ffmpeg seeks to the point rather than decoding and
        # discarding everything before it.
        "-ss",
        str(start),
        "-t",
        str(length),
        "-i",
        str(src),
        "-af",
        f"volume={gain_db:.4f}dB",
        "-ac",
        "1",
        "-c:a",
        "libopus",
        "-b:a",
        "64k",
        # Without these the WebM muxer writes random track/segment UIDs and the
        # encoder tag, so the same WAV encoded twice gave different files.
        "-fflags",
        "+bitexact",
        "-flags:a",
        "+bitexact",
        str(dest),
    ]


def corpus_gain_db(peaks: list[float], target_tp: float = TARGET_TP) -> float:
    """The ONE gain, in dB, applied to every file in the corpus.

    Chosen so the LOUDEST window in the whole tree lands at `target_tp` and
    every other file keeps its distance below it. That distance is the point:
    it is the game's own mix, and per-file normalisation destroys it.

    Pure, so the choice is testable without ffmpeg. `-inf` peaks (a handful of
    events render to digital silence) carry no level information and are
    skipped rather than treated as quiet -- `max()` would ignore them anyway,
    but an all-silent corpus must come out 0.0 rather than raising or
    producing infinite gain.
    """
    finite = [p for p in peaks if p != float("-inf")]
    if not finite:
        return 0.0
    return target_tp - max(finite)


def window_peak_args(src: Path, start: float, length: float) -> list[str]:
    """ffmpeg invocation whose stderr carries the peak level of one window.

    Measures the SAME window `ffmpeg_args` will encode -- not the whole file.
    Trimming happens first, so a lead-in or a long silent tail that gets cut
    must not influence the level the corpus gain is computed from.
    """
    return [
        "ffmpeg",
        "-hide_banner",
        "-ss",
        str(start),
        "-t",
        str(length),
        "-i",
        str(src),
        "-af",
        "astats=metadata=0",
        "-f",
        "null",
        "-",
    ]


def measure_window_peak_dbfs(src: Path, start: float, length: float) -> float:
    out = subprocess.run(window_peak_args(src, start, length), capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg astats failed on {src} (exit {out.returncode}):\n{out.stderr}")
    try:
        return parse_peak_dbfs(out.stderr)
    except ValueError as e:
        raise ValueError(f"{e} (file: {src})") from e


def parse_peak_dbfs(astats_output: str) -> float:
    """Pure parser: the max `Peak level dB` line from ffmpeg's `astats` filter.

    Testable without invoking ffmpeg — mirrors `trim_window`'s split between
    the ffmpeg invocation and its output parsing.

    A handful of the 566 rendered WAVs are pure digital silence (a Task 2
    render artifact, not a Task 3 concern) and ffmpeg reports those as `-inf`,
    not a number — `float("-inf")` parses that correctly, but only if the
    pattern below actually matches the literal `inf`.
    """
    peaks = [float(m) for m in re.findall(r"Peak level dB:\s*(-?(?:inf|[\d.]+))", astats_output)]
    if not peaks:
        raise ValueError("astats output carried no 'Peak level dB' reading")
    return max(peaks)


def measure_peak_dbfs(path: Path) -> float:
    """Actual peak level, in dBFS, of the rendered file at *path*.

    Measures the real decoded output, not the pre-encode PCM: this is the
    check that reflects what a browser will actually play, which is the
    thing TARGET_TP alone cannot guarantee.
    """
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "astats=metadata=0", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg astats failed on {path} (exit {out.returncode}):\n{out.stderr}")
    try:
        return parse_peak_dbfs(out.stderr)
    except ValueError as e:
        raise ValueError(f"{e} (file: {path})") from e


def verify_peaks(files: list[Path], limit: float = PEAK_LIMIT_DBFS) -> list[tuple[Path, float]]:
    """Fail loudly, naming every offender, if any file in *files* exceeds *limit* dBFS.

    Runs against the finished .webm, not an intermediate buffer — that
    mismatch (asserting a property of the pre-encode PCM, then reporting it as
    a property of the shipped artifact) is exactly how the clipping this
    guards against went undetected in the first place.

    Returns every (file, peak) pair measured, so a caller can also spot other
    things worth a human ear — e.g. a file that measured pure silence — without
    a second ffmpeg pass over the same files.
    """
    measured = [(f, measure_peak_dbfs(f)) for f in files]
    over = [(str(f), peak) for f, peak in measured if peak > limit]
    if over:
        detail = "\n".join(f"  {name}: {peak:+.2f} dBFS" for name, peak in over)
        raise RuntimeError(
            f"{len(over)} encoded file(s) exceed the {limit:+.1f} dBFS peak limit "
            f"(Opus reconstruction can overshoot loudnorm's pre-encode TP target):\n{detail}"
        )
    return measured


def _duration(src: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(src),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())


# What `encode_tree` treats as a source cue. `.wav` is what `extract_sfx`'s
# FMOD renderer writes; `.mp3` is what it copies verbatim out of
# `res://debug_audio/` (the game's non-FMOD sound layer — see
# `extract_sfx.extract_debug_audio`). Both go through the identical trim /
# measure / single-corpus-gain path, which is the point: the debug-audio cues
# and the FMOD cues are mixed against each other in the game, so normalising
# them separately would put them at the wrong relative level.
SOURCE_SUFFIXES = (".wav", ".mp3")


def source_files(src_root: Path) -> list[Path]:
    """Every encodable source cue under *src_root*, in a stable order."""
    return sorted(p for p in src_root.rglob("*") if p.suffix.lower() in SOURCE_SUFFIXES)


def encode_tree(src_root: Path, dest_root: Path) -> list[Path]:
    """Trim then encode every source cue under *src_root* to a .webm under *dest_root*.

    Two passes, and the split is not incidental: the gain applied at encode
    time depends on the LOUDEST window in the whole tree (`corpus_gain_db`),
    so every window has to be measured before any file can be encoded. Doing
    it per-file instead is what produced the flattened, un-gamelike mix this
    replaced.

    Logs each file detected as a loop, each file whose window still hit the
    hard cap despite that (never falls silent AND isn't a detected loop), and
    each file that came out pure silence — all three are worth a human ear
    before shipping. Also prints a progress line every `PROGRESS_EVERY` files,
    since this walks hundreds of files through several ffmpeg passes each and
    can run for a while.
    """
    wavs = sorted(source_files(src_root))
    total = len(wavs)
    written: list[Path] = []
    capped: list[str] = []
    loops: list[tuple[str, float]] = []
    plans: list[tuple[Path, Path, str, float, float]] = []
    peaks: list[float] = []
    print(f"measuring trim windows and peak levels on {total} file(s)...", flush=True)
    for i, wav in enumerate(wavs, start=1):
        dest = dest_root / wav.relative_to(src_root).with_suffix(".webm")
        detect = subprocess.run(detect_args(wav), capture_output=True, text=True)
        duration = _duration(wav)
        rel = wav.relative_to(src_root).as_posix()
        cap = cap_for(rel)
        # Precedence: lead-in strip (trim_window's `start`) -> loop detection
        # -> silence gap -> cap_for(). A looping file has no silence gap to
        # find — the ~95 that hit the cap under gap/cap-only logic were
        # exactly the loops, shipping ~20 repetitions of "one gold coin" as
        # 8 s of coins. Loop detection runs on every file (not just the
        # capped ones) and only overrides trim_window's gap/cap-based length
        # when it actually finds a strong periodic signal.
        start, length = trim_window(detect.stderr, duration, cap)
        period = detect_loop(_decode_samples(wav, start), LOOP_SR)
        if period is not None:
            length = round(min(period, cap), 3)
            loops.append((rel, period))
        if length >= cap:
            capped.append(rel)
        plans.append((wav, dest, rel, start, length))
        peaks.append(measure_window_peak_dbfs(wav, start, length))
        if i % PROGRESS_EVERY == 0 or i == total:
            print(f"[{i}/{total}] measured {rel}", flush=True)

    gain_db = corpus_gain_db(peaks)
    finite = [p for p in peaks if p != float("-inf")]
    print(
        f"-- corpus gain {gain_db:+.2f} dB "
        f"(loudest window {max(finite, default=float('-inf')):+.2f} dBFS, "
        f"quietest {min(finite, default=float('-inf')):+.2f} dBFS, "
        f"dynamic range preserved: "
        f"{(max(finite, default=0.0) - min(finite, default=0.0)):.1f} dB)",
        flush=True,
    )

    for i, (wav, dest, rel, start, length) in enumerate(plans, start=1):
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Encode to a per-process temp path, then atomically rename into
        # place. Without this, an interrupted run (or a second process
        # writing the same tree concurrently — this has actually happened;
        # see the encode-sfx recipe's `set -o pipefail` sibling fix) can
        # leave a partially-written .webm that a file *count* sails right
        # past — os.replace is atomic on the same filesystem, so `dest`
        # either has the old complete file, the new complete file, or (only
        # before the first successful encode) doesn't exist yet; never a
        # partial one. The PID in the name keeps two concurrent invocations
        # from racing on the same temp file.
        # Keep the .webm extension on the temp name (not just a bare `.tmp`
        # suffix) — ffmpeg infers the output muxer from the extension when no
        # `-f` is given, so a temp name ending in anything else makes it fail
        # immediately with "Unable to find a suitable output format".
        tmp_dest = dest.with_name(f".{dest.stem}.{os.getpid()}.tmp.webm")
        # ffmpeg crashing, the process getting killed, the disk filling, a
        # source file dying mid-decode, or a human Ctrl-C'ing a long run (the
        # single most likely interruption here, since this loop is exactly
        # the kind of long job a human tails and may stop) can all leave
        # tmp_dest behind half-written. That file is still inside dest_root
        # (shipped site/public/audio/), just hidden behind its leading dot,
        # so leaving it there would trade the corruption hazard at `dest`
        # (what os.replace's atomicity prevents) for the identical hazard one
        # path segment over. `finally` + a success flag (rather than
        # `except Exception`) is deliberate: KeyboardInterrupt and SystemExit
        # derive from BaseException, not Exception, so an `except Exception`
        # here would silently skip cleanup on Ctrl-C — the case that matters
        # most. Unlink unconditionally whenever the rename didn't happen,
        # tolerating the file not existing (e.g. ffmpeg failed before writing
        # anything); nothing here is caught or swallowed, so every exception
        # still propagates unchanged.
        replaced = False
        try:
            subprocess.run(
                ffmpeg_args(wav, tmp_dest, start, length, gain_db),
                check=True,
                capture_output=True,
            )
            os.replace(tmp_dest, dest)
            replaced = True
        finally:
            if not replaced:
                tmp_dest.unlink(missing_ok=True)
        written.append(dest)
        if i % PROGRESS_EVERY == 0 or i == total:
            # flush=True: stdout is line-buffered only for a tty. Piped through
            # `tee` to a log file (the incremental-progress requirement this
            # loop exists for), Python defaults to block-buffering, so without
            # this a human tailing the log sees nothing until the run exits —
            # indistinguishable from "stuck".
            print(f"[{i}/{total}] encoded {rel}", flush=True)
    if loops:
        periods = sorted(p for _, p in loops)
        mid = periods[len(periods) // 2]
        print(
            f"-- {len(loops)} file(s) detected as loops "
            f"(period min={periods[0]:.3f}s median={mid:.3f}s max={periods[-1]:.3f}s):"
        )
        for rel, period in loops:
            print(f"     {rel}: {period:.3f}s")
    if capped:
        print(f"!! {len(capped)} file(s) hit their cap (never fall silent):")
        for rel in capped:
            print(f"     {rel}")
    print(
        f"verifying peak levels on {total} file(s) (limit {PEAK_LIMIT_DBFS:+.1f} dBFS)...",
        flush=True,
    )
    measured = verify_peaks(written)
    print("all files within the peak limit", flush=True)
    # Backstop for a failure path the per-file try/except didn't anticipate:
    # if any temp file is still here, something is wrong even though the run
    # otherwise "succeeded" (every wav got a written .webm).
    leftover_tmp = sorted(dest_root.rglob("*.tmp.webm"))
    if leftover_tmp:
        names = ", ".join(str(p.relative_to(dest_root)) for p in leftover_tmp)
        raise RuntimeError(f"{len(leftover_tmp)} .tmp file(s) left under {dest_root}: {names}")
    # Pure digital silence (peak == -inf) trivially passes the limit but is
    # almost certainly a Task 2 render artifact (the FMOD event produced no
    # audio at all), not a usable cue — worth a human ear same as `capped`.
    silent = [str(f.relative_to(dest_root)) for f, peak in measured if peak == float("-inf")]
    if silent:
        print(f"!! {len(silent)} file(s) encoded as pure silence:")
        for rel in silent:
            print(f"     {rel}")
    return written


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="extracted/<version>/audio")
    ap.add_argument("dest", type=Path, help="site/public/audio")
    args = ap.parse_args()
    files = encode_tree(args.src, args.dest)
    total = sum(p.stat().st_size for p in files)
    print(f"encoded {len(files)} files, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
