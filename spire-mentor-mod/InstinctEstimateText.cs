// InstinctEstimateText.cs -- Instinct's win estimate (`advice.estimate`,
// PROTOCOL.md section 5; GH #1222) in words, for the advice panel's headline.
// Game-free so tests/unit can pin the copy; SimHud and the frame only lay it
// out.
//
// Every number assumes Instinct plays every move from here: it comes from a
// small net reading the student's own view of the position, not a search. The
// bands match the workbench's top-bar chip (web/src/advice/estimate.ts
// estimateChip): a run is good from 60%, bad under 30%; a fight good from 90%,
// bad under 60% (most hallway fights sit near 100%, so under 90% is news).

using System;
using System.Globalization;
using System.Text.Json;

namespace SpireMentor;

public enum EstimateTone
{
    Good,
    Mid,
    Bad,
}

/// <summary>`advice.estimate`: P(win the run), P(win this fight) and the
/// fight's HP-cost quantiles (null outside a fight).</summary>
public sealed record InstinctEstimate(double PRun, double? PFight, double? HpQ10, double? HpQ50, double? HpQ90);

internal static class InstinctEstimateText
{
    /// <summary>The dim line under the headline: what the numbers mean.</summary>
    public const string Caption = "if Instinct plays every move from here";

    /// <summary>`advice.estimate`, or null when absent or malformed (older
    /// advisors, weights without a probe, a forced step).</summary>
    public static InstinctEstimate? Parse(JsonElement advice)
    {
        if (advice.ValueKind != JsonValueKind.Object ||
            !advice.TryGetProperty("estimate", out JsonElement e) ||
            e.ValueKind != JsonValueKind.Object ||
            Num(e, "p_run") is not double pRun)
        {
            return null;
        }
        double? q10 = null, q50 = null, q90 = null;
        if (e.TryGetProperty("hp_loss", out JsonElement hp) && hp.ValueKind == JsonValueKind.Object)
        {
            q10 = Num(hp, "q10");
            q50 = Num(hp, "q50");
            q90 = Num(hp, "q90");
        }
        return new InstinctEstimate(pRun, Num(e, "p_fight"), q10, q50, q90);
    }

    private static double? Num(JsonElement o, string name) =>
        o.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;

    /// <summary>"71%": rounded, clamped to 0..100.</summary>
    public static string Pct(double p) =>
        Math.Round(Math.Clamp(p, 0.0, 1.0) * 100, MidpointRounding.AwayFromZero).ToString("0", CultureInfo.InvariantCulture) + "%";

    public static EstimateTone RunTone(double p) => Band(p, 0.6, 0.3);

    public static EstimateTone FightTone(double p) => Band(p, 0.9, 0.6);

    private static EstimateTone Band(double p, double good, double bad) =>
        p >= good ? EstimateTone.Good : p < bad ? EstimateTone.Bad : EstimateTone.Mid;

    /// <summary>"Run 71% · Fight 100%"; the fight only in a fight.</summary>
    public static string Headline(InstinctEstimate e) =>
        e.PFight is double f ? $"Run {Pct(e.PRun)} · Fight {Pct(f)}" : $"Run {Pct(e.PRun)}";

    /// <summary>"−9 HP", "+3 HP" (a heal), "±0 HP": the fight's HP change,
    /// signed from the player's side.</summary>
    public static string HpDelta(double loss)
    {
        int v = (int)Math.Round(loss, MidpointRounding.AwayFromZero);
        return v == 0 ? "±0 HP" : v > 0 ? $"−{v} HP" : $"+{-v} HP";
    }

    /// <summary>"This fight: −9 HP (−2 HP to −20 HP)", or null outside a fight.
    /// The range is the 10th to 90th percentile; it is left out when both ends
    /// round to the same number.</summary>
    public static string? HpLine(InstinctEstimate e)
    {
        if (e.HpQ50 is not double mid)
        {
            return null;
        }
        string text = $"This fight: {HpDelta(mid)}";
        if (e.HpQ10 is double lo && e.HpQ90 is double hi && HpDelta(lo) != HpDelta(hi))
        {
            text += $" ({HpDelta(lo)} to {HpDelta(hi)})";
        }
        return text;
    }
}
