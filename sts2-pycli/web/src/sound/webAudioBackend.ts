import type { AudioBackend } from "./player";

/**
 * The real WebAudio implementation. Kept deliberately thin — everything
 * decision-shaped lives in `player.ts`, which is unit-tested.
 *
 * The AudioContext is constructed here. Sound is on by default, so this
 * usually runs at page load with NO user gesture behind it, and browsers then
 * hand back a context in the `suspended` state (Chrome's autoplay policy;
 * Safari's is stricter still). A suspended context accepts `start()` calls
 * and plays nothing, forever — the old code created it only from the enable
 * click and so never had to care, but a returning visitor's stored "on"
 * re-created it at mount and heard nothing until they cleared storage. So:
 * the first pointer / key gesture anywhere on the page resumes it, and a
 * `play()` that finds it still suspended asks again (a play inside a click
 * handler is itself a gesture).
 */
export function createWebAudioBackend(): AudioBackend {
  const ctx = new AudioContext();
  let voices = 0;

  const GESTURES = ["pointerdown", "keydown", "touchend"] as const;
  const unhook = () => {
    for (const ev of GESTURES) window.removeEventListener(ev, resume, true);
  };
  const resume = () => {
    if (ctx.state === "running") {
      unhook();
      return;
    }
    ctx.resume().then(unhook, () => { /* still blocked: keep listening */ });
  };
  if (ctx.state !== "running" && typeof window !== "undefined") {
    // Capture phase, so a handler that stops propagation (a menu closing on
    // pointerdown) cannot starve the resume of the gesture it needs.
    for (const ev of GESTURES) window.addEventListener(ev, resume, { capture: true, passive: true });
  }

  return {
    async load(url: string) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${url}: ${res.status}`);
      return ctx.decodeAudioData(await res.arrayBuffer());
    },
    play(buffer, gain, rate) {
      if (ctx.state === "suspended") resume();
      const src = ctx.createBufferSource();
      src.buffer = buffer as AudioBuffer;
      // Godot's `PitchScale` resamples (pitch and speed together), which is
      // exactly what `playbackRate` does — not a pitch-preserving shift.
      src.playbackRate.value = rate;
      const g = ctx.createGain();
      g.gain.value = gain;
      src.connect(g).connect(ctx.destination);
      voices += 1;
      src.onended = () => { voices -= 1; };
      try {
        src.start();
      } catch (err) {
        // `start()` can throw synchronously (an already-started node, a
        // closed context, an invalid buffer). `onended` only ever fires for
        // a node that actually started, so without this the increment above
        // would never be undone — one bad `play()` call would permanently
        // inflate `voices`, and enough of them would pin the count at
        // `maxVoices` and silence every future cue for the rest of the tab's
        // life. Undo the increment on the same path that failed to start.
        console.warn("[sound] AudioBufferSourceNode.start() threw; unwound the voice count", err);
        voices -= 1;
        src.onended = null;
      }
    },
    activeVoices: () => voices,
  };
}
