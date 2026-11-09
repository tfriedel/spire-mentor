// CoachStyle.cs -- the advice STYLE inside Advisor mode (coach plan 3):
//
//   Hints  the AI's opinion is visible BEFORE you act (ranked panel, eval
//          bar, card scores, map markers) -- the mod's behaviour before v5.
//   Coach  nothing before you act; F5 peeks at the next pick only, and the
//          coach chip says afterwards where you diverged from the AI.
//
// Mode Off / Autopilot are unchanged. Deliberately free of game types so
// tests/unit can compile it on its own; SimRuntime holds the live value.
//
// The badge's Mode switch (SimToolbar) folds the three runtime knobs a player
// thinks of as one choice -- SimMode, the style, and F7 autoplay -- into
// Off / Coach / Hints / AI plays (MentorMode, MentorModes.Plan below). "AI
// plays" is F7: the AI plays THIS run in Advisor mode until stopped, handing
// back what it cannot play (`pause_on`). It is NOT SimMode.Autopilot, which an
// external agent drives (menus and new runs included) and which only the
// config file can arm at launch; while that holds the run, the fourth segment
// reads "Autopilot" and the other three take the controls back, as F8 does.

using System;

namespace SpireMentor;

public enum AdviceStyle
{
    Hints,
    Coach,
}

/// <summary>Game-free mirror of <c>SimMode</c> (same order, same values),
/// so the cycle can be unit-tested without the game assemblies.</summary>
public enum SimModeLite
{
    Off,
    Advisor,
    Autopilot,
}

internal static class CoachStyle
{
    /// <summary>The persisted style flag for a fresh config: Hints (the AI's
    /// opinion while you decide). After that the style switched last is kept
    /// (SimActions.SwitchStyle saves it).</summary>
    public const bool CoachByDefault = false;

    /// <summary>
    /// F8 / the on-off button: AI off &lt;-&gt; on, in whatever style is set.
    /// From Autopilot the caller stops the AI first and lands in Advisor.
    /// The style is never touched here -- switching Hints/Coach is its own
    /// control (<see cref="SwitchStyle"/>), so changing style can never turn
    /// the AI off.
    /// </summary>
    public static (SimModeLite mode, AdviceStyle style) ToggleOnOff(SimModeLite mode, AdviceStyle style) =>
        mode == SimModeLite.Advisor ? (SimModeLite.Off, style) : (SimModeLite.Advisor, style);

    /// <summary>The style button: Hints &lt;-&gt; Coach. The mode is left as it
    /// is (with the AI off it just picks the style the next F8 turns on in).</summary>
    public static (SimModeLite mode, AdviceStyle style) SwitchStyle(SimModeLite mode, AdviceStyle style) =>
        (mode, style == AdviceStyle.Coach ? AdviceStyle.Hints : AdviceStyle.Coach);

    /// <summary>The toolbar's on/off button.</summary>
    public static string PowerLabel(SimModeLite mode) =>
        mode switch
        {
            SimModeLite.Advisor => "AI on",
            SimModeLite.Autopilot => "Take over",
            _ => "AI off",
        };

    /// <summary>The toolbar's style button.</summary>
    public static string StyleLabel(AdviceStyle style) => style == AdviceStyle.Coach ? "Coach" : "Hints";

    /// <summary>What the mode is called in prose (pill, help, log lines).</summary>
    public static string ModeName(SimModeLite mode, AdviceStyle style) =>
        mode switch
        {
            SimModeLite.Advisor => style == AdviceStyle.Coach ? "Coach" : "Hints",
            SimModeLite.Autopilot => "Autopilot",
            _ => "AI off",
        };

    /// <summary>The persisted tickbox ("start in Coach style") as a style.</summary>
    public static AdviceStyle FromDefaultFlag(bool coach) => coach ? AdviceStyle.Coach : AdviceStyle.Hints;

    /// <summary>`hints` / `coach`, case-insensitive; null for anything else.</summary>
    public static AdviceStyle? Parse(string? text) =>
        (text ?? "").Trim().ToLowerInvariant() switch
        {
            "hints" or "hint" => AdviceStyle.Hints,
            "coach" => AdviceStyle.Coach,
            _ => null,
        };
}

/// <summary>The badge's Mode switch: what the player picks, one of four.
/// Autobattle is not a fifth: it is a way of playing in Coach or Hints (the
/// run-level decisions stay the player's), so it shows under its style.</summary>
public enum MentorMode
{
    Off,
    Coach,
    Hints,
    /// <summary>F7 autoplay (or, read-only, a config-armed Autopilot).</summary>
    AiPlays,
}

/// <summary>
/// What picking a <see cref="MentorMode"/> does, as steps for SimActions: stop
/// a config-armed Autopilot first, then land in <c>Mode</c> / <c>Style</c>, then
/// switch autobattle off and plain autoplay on or off.
/// </summary>
public readonly record struct MentorModePlan(
    SimModeLite Mode,
    AdviceStyle Style,
    bool StopAutopilot,
    bool StopAutobattle,
    bool ArmAutoplay,
    bool DisarmAutoplay);

internal static class MentorModes
{
    /// <summary>Left to right, as the switch shows them.</summary>
    public static readonly MentorMode[] All =
        { MentorMode.Off, MentorMode.Coach, MentorMode.Hints, MentorMode.AiPlays };

    /// <summary>
    /// The segment lit for the runtime state. F7 autoplay is "AI plays", and
    /// so is Autopilot; autobattle keeps its style lit (the player still makes
    /// the run's decisions; the Autoplay row says autobattle is on).
    /// </summary>
    public static MentorMode Current(SimModeLite mode, AdviceStyle style, bool autoplayArmed, bool autobattleOn) =>
        mode switch
        {
            SimModeLite.Off => MentorMode.Off,
            SimModeLite.Autopilot => MentorMode.AiPlays,
            _ when autoplayArmed && !autobattleOn => MentorMode.AiPlays,
            _ => style == AdviceStyle.Coach ? MentorMode.Coach : MentorMode.Hints,
        };

    /// <summary>
    /// The steps that take the runtime state to <paramref name="target"/>, or
    /// null when it is already there: the lit segment does nothing, and "AI
    /// plays" under Autopilot is already the AI playing.
    /// </summary>
    public static MentorModePlan? Plan(MentorMode target, SimModeLite mode, AdviceStyle style,
        bool autoplayArmed, bool autobattleOn)
    {
        if (target == Current(mode, style, autoplayArmed, autobattleOn))
        {
            return null;
        }
        bool fromAutopilot = mode == SimModeLite.Autopilot;
        return target switch
        {
            // Leaving Advisor mode disarms autoplay (and autobattle) by itself.
            MentorMode.Off => new MentorModePlan(SimModeLite.Off, style,
                StopAutopilot: fromAutopilot, StopAutobattle: false, ArmAutoplay: false, DisarmAutoplay: false),
            MentorMode.Coach or MentorMode.Hints => new MentorModePlan(
                SimModeLite.Advisor,
                target == MentorMode.Coach ? AdviceStyle.Coach : AdviceStyle.Hints,
                StopAutopilot: fromAutopilot,
                // A style switch under autobattle keeps the autobattle.
                StopAutobattle: false,
                ArmAutoplay: false,
                DisarmAutoplay: mode == SimModeLite.Advisor && autoplayArmed && !autobattleOn),
            // From Off it turns the coach on in the style that is set, then
            // arms F7; from autobattle it trades the autobattle for F7.
            _ => new MentorModePlan(SimModeLite.Advisor, style,
                StopAutopilot: false, StopAutobattle: autobattleOn, ArmAutoplay: true, DisarmAutoplay: false),
        };
    }

    /// <summary>A segment's word. The fourth reads "Autopilot" while the
    /// config-armed Autopilot holds the run, so it never claims to be F7.</summary>
    public static string Label(MentorMode mode, bool autopilot) =>
        mode switch
        {
            MentorMode.Off => "Off",
            MentorMode.Coach => "Coach",
            MentorMode.Hints => "Hints",
            _ => autopilot ? "Autopilot" : "AI plays",
        };
}
