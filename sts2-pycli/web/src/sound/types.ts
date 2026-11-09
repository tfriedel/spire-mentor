/** Whether a cue carries information or is texture. `reduced` mode drops chatter. */
export type CueClass = "informative" | "chatter";

/** One sound to play, named by its FMOD event path. */
export interface SoundCue {
  event: string;
  cls: CueClass;
  /** 0..1 multiplier applied on top of the user's volume. Defaults to 1.
   *  For a `res://debug_audio/` cue this is the `volume` argument of the
   *  game's own `NDebugAudioManager.Play` call (`card_deal` plays at 0.25,
   *  the combat-start potion slosh at 0.3). */
  gain?: number;
  /** Random pitch spread, as the fraction `NDebugAudioManager
   *  .GetRandomPitchScale` draws from: each play runs at a playback rate
   *  uniform in `1 ± pitchVariance` (`PitchVariance.Small` = 0.02, `Medium`
   *  = 0.05, `Large` = 0.1). Like Godot's `PitchScale`, the rate changes
   *  pitch and speed together. Absent = 0 = the file exactly as rendered. */
  pitchVariance?: number;
}

/** The game's `PitchVariance` enum -> its spread, from
 *  `NDebugAudioManager.GetRandomPitchScale`. */
export const PITCH_VARIANCE = { none: 0, small: 0.02, medium: 0.05, large: 0.1 } as const;
