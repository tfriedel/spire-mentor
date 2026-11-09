// SpeedSteps.cs -- the speed hotkeys' steps and autobattle's speed rule.
//
// F9 / F10 and the toolbar's -/+ step through 1x, 2x, 3x, 4x, 6x, 8x:
// tenths were too fine to be worth a key press. The settings slider and the
// `spirementor speed` console command still set any speed SimSpeed allows.
//
// Autobattle (Autobattle.cs) at 1x means watching every fight at normal
// speed, so switching it on raises 1x to 2x. Any other speed is the player's
// own choice and is left alone; switching autobattle off puts 1x back only
// if the speed is still the 2x autobattle set.
//
// Game-free on purpose (no Godot, no sts2.dll) so tests/unit can link it;
// SimSpeed applies the result.

using System;

namespace SpireMentor;

internal static class SpeedSteps
{
    /// <summary>
    /// The speeds F9 / F10 and the toolbar's -/+ visit. Whole speeds to 4x,
    /// then 6x and 8x: past 4x a +1 step is too small to notice.
    /// </summary>
    public static readonly double[] Steps = { 1.0, 2.0, 3.0, 4.0, 6.0, 8.0 };

    /// <summary>The slowest and fastest step (SimSpeed.Max is 8).</summary>
    public const double MinStep = 1.0;
    public const double MaxStep = 8.0;

    /// <summary>The speed autobattle raises 1x to.</summary>
    public const double Autobattle = 2.0;

    private const double Eps = 1e-6;

    /// <summary>
    /// The speed one step up (<paramref name="direction"/> &gt; 0) or down
    /// from <paramref name="current"/>: the next of <see cref="Steps"/> in
    /// that direction. A speed between steps (1.5x or 5x from the slider)
    /// goes to the nearest one in that direction (2x up, 1x down). A speed
    /// below 1x is never sped up by "down"; "up" takes it to 1x.
    /// </summary>
    public static double Step(double current, int direction)
    {
        if (direction > 0)
        {
            foreach (double s in Steps)
            {
                if (s > current + Eps)
                {
                    return s;
                }
            }
            return MaxStep;
        }
        if (current <= MinStep + Eps)
        {
            return current;
        }
        for (int i = Steps.Length - 1; i >= 0; i--)
        {
            if (Steps[i] < current - Eps)
            {
                return Steps[i];
            }
        }
        return MinStep;
    }

    /// <summary>The speed to switch to as autobattle turns on, or null to
    /// leave <paramref name="current"/> alone (anything but 1x).</summary>
    public static double? AutobattleOn(double current) =>
        Math.Abs(current - 1.0) < Eps ? Autobattle : null;

    /// <summary>As autobattle turns off, having raised the speed: whether to
    /// put 1x back, i.e. the player has not changed it since.</summary>
    public static bool RestoreOnOff(double current) => Math.Abs(current - Autobattle) < Eps;

    /// <summary>The speed as the toolbar shows it: "2x", "1.5x".</summary>
    public static string Label(double speed) =>
        speed.ToString("0.#", System.Globalization.CultureInfo.InvariantCulture) + "x";
}
