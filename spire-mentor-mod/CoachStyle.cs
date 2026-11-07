// CoachStyle.cs -- the advice STYLE inside Advisor mode (coach plan 3):
//
//   Hints  the AI's opinion is visible BEFORE you act (ranked panel, eval
//          bar, card scores, map markers) -- the mod's behaviour before v5.
//   Coach  nothing before you act; F5 peeks at the next pick only, and the
//          coach chip says afterwards where you diverged from the AI.
//
// Mode Off / Autopilot are unchanged. Deliberately free of game types so
// tests/unit can compile it on its own; SimRuntime holds the live value.

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
