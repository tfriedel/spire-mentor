import { eventAssetUrl } from "./sfxMap";
import type { SoundCue } from "./types";

/** The WebAudio calls, behind an interface so the policy above can be tested. */
export interface AudioBackend {
  load(url: string): Promise<unknown>;
  /** `rate` is the playback rate, 1 = as rendered (see `SoundCue.pitchVariance`). */
  play(buffer: unknown, gain: number, rate: number): void;
  activeVoices(): number;
}

/**
 * Decides *whether* to play a cue: buffer caching, a voice cap so a burst
 * cannot stack, and a per-event debounce. `now` is injected rather than read
 * from the clock so the policy is testable without timers.
 */
export class SoundPlayer {
  private readonly buffers = new Map<string, Promise<unknown>>();
  private readonly lastPlayed = new Map<string, number>();
  private readonly maxVoices: number;
  private readonly debounceMs: number;
  private readonly rand: () => number;

  constructor(
    private readonly backend: AudioBackend,
    opts: { maxVoices?: number; debounceMs?: number; rand?: () => number } = {},
  ) {
    // Recounted (2026-08-24) when the cue set roughly doubled — turn banners,
    // card deal/exhaust, the orb table, the universal monster death fade and
    // the per-card hit layer all landed at once. A single round boundary in a
    // Defect AoE turn can now legitimately want ~15 distinct sounds, and the
    // old cap of 8 silently dropped everything past the eighth: the cap is
    // there to stop one event MACHINE-GUNNING (which `cuesForCombat`'s `seen`
    // set and the per-event debounce below already prevent), not to bound how
    // many different things may happen at once.
    this.maxVoices = opts.maxVoices ?? 16;
    this.debounceMs = opts.debounceMs ?? 60;
    // Uniform [0,1), injected so the pitch draw is testable. The game draws
    // from `Rng.Chaotic`, a stream nothing else observes, so any uniform
    // source reproduces it.
    this.rand = opts.rand ?? Math.random;
  }

  /** `NDebugAudioManager.GetRandomPitchScale`: `1 + NextFloat(-v, v)`, and
   *  exactly 1, with no draw, when the call site asked for no variance. */
  private rate(variance: number | undefined): number {
    if (!variance) return 1;
    return 1 + (this.rand() * 2 - 1) * variance;
  }

  async play(cue: SoundCue, volume: number, now: number): Promise<void> {
    const last = this.lastPlayed.get(cue.event);
    if (last !== undefined && now - last < this.debounceMs) return;
    // Checked, and dropped, BEFORE the debounce window is stamped: a cue the
    // voice cap kills never played, so it must not suppress the next attempt
    // at (or near) the same `now` once a voice frees up.
    if (this.backend.activeVoices() >= this.maxVoices) return;
    this.lastPlayed.set(cue.event, now);

    let buffer = this.buffers.get(cue.event);
    if (!buffer) {
      buffer = this.backend.load(eventAssetUrl(cue.event));
      this.buffers.set(cue.event, buffer);
    }
    try {
      this.backend.play(await buffer, volume * (cue.gain ?? 1), this.rate(cue.pitchVariance));
    } catch (err) {
      // A missing or undecodable file must never break the frame that
      // triggered it. This also catches a throw from `backend.play` itself,
      // not only `backend.load` -- we can't tell which without inspecting
      // `err` in more depth than is worth doing here, so the message names
      // both without asserting which one is at fault. Evicting the cache
      // entry is harmless and idempotent either way: it just means the next
      // call re-`load`s, which is exactly what should happen so a later
      // deploy of the asset (or a transient backend hiccup) can recover.
      console.warn(
        `[sound] ${cue.event} (${eventAssetUrl(cue.event)}) failed to load or play` +
          " -- evicting the cached buffer so a later attempt can retry",
        err,
      );
      this.buffers.delete(cue.event);
    }
  }
}
