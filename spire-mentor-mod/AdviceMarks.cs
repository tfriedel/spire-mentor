// AdviceMarks.cs -- what the HUD highlights, decided without the game.
//
// The advice is a ranked list of moves (`options[]`) plus, for a fight, the
// turn's play order (`plan[]`, PROTOCOL.md section 5). The HUD turns that
// into marks on the things the player will click: a ring and a step number
// on each card to play, the End Turn button, a potion slot, a reward card, a
// shop slot, a campfire button. This file decides WHICH things get a mark
// and how loud it is; SimHighlights draws them.
//
// Rules (reported 2026-11-07 after the first hand-played Instinct run):
//   * one mark per thing: a targeted card has one option PER TARGET, and
//     drawing a label for each put two numbers on top of each other on one
//     card. The card's mark is its best option's; the target travels with it;
//   * a move nobody would make (under 0.5% of the search, or of the net's
//     probability) gets no mark at all -- "0%" over a card is noise;
//   * the plan numbers the cards 1, 2, 3 in the order to play them; without a
//     plan only the pick is numbered;
//   * the words a row is ranked with are a player's ("Best", "Also good",
//     "Okay", "Weak", "Mistake"), not the engine's units, which stay behind the
//     "Show numbers" setting.
//
// Game-free (no Godot, no sts2.dll) so tests/unit can compile it.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace SpireMentor;

/// <summary>How loud a mark is.</summary>
public enum MarkTier
{
    /// <summary>Not marked.</summary>
    None,

    /// <summary>A real alternative: a thin ring, no glow.</summary>
    Alt,

    /// <summary>Part of the plan after its first step, or as good as the pick.</summary>
    Good,

    /// <summary>The move to make now: gold, glowing, numbered 1.</summary>
    Best,
}

/// <summary>What a mark is drawn on.</summary>
public enum MarkKind
{
    Card,
    Potion,
    EndTurn,
    ScreenOption,
    Skip,
}

/// <summary>The part of an advice option the marks need (SimAdviceOption, game-free).</summary>
public sealed record MarkOption(
    string Label,
    string Action,
    int? CardIndex,
    int? TargetIndex,
    int? AllyIndex,
    int? Slot,
    int? Index,
    bool IsBest,
    bool InBand,
    double? Share,
    double? Q,
    int? PickRank);

/// <summary>One step of the turn's plan (`advice.plan[]`).</summary>
public sealed record MarkPlanStep(
    string Action,
    int? CardIndex,
    int? TargetIndex,
    int? AllyIndex,
    int? Slot,
    string Label);

/// <summary>One thing to highlight.</summary>
public sealed record Mark(
    MarkKind Kind,
    int Key,
    MarkTier Tier,
    int? Order,
    int? TargetIndex,
    int? AllyIndex,
    string Label,
    double? Share,
    double? Q);

public static class AdviceMarks
{
    /// <summary>Below this share / probability a move is not shown at all.</summary>
    public const double HiddenShare = 0.005;

    /// <summary>From this share a non-plan move is worth a thin ring.</summary>
    public const double AltShare = 0.15;

    /// <summary>A Q this close to the pick's is "as good" (search only).</summary>
    public const double GoodQDelta = 0.005;

    /// <summary>Most rings drawn for alternatives, so a wide hand stays readable.</summary>
    public const int MaxAlts = 2;

    /// <summary>The thing an option acts on, or null when it has none on screen.</summary>
    public static (MarkKind Kind, int Key)? Target(MarkOption o)
    {
        switch (o.Action)
        {
            case "end_turn":
                return (MarkKind.EndTurn, 0);
            case "potion":
                return o.Slot is int slot && slot >= 0 ? (MarkKind.Potion, slot) : null;
            case "skip":
                return (MarkKind.Skip, 0);
            case "choose":
                return o.Index is int idx && idx >= 0 ? (MarkKind.ScreenOption, idx) : null;
        }
        if (o.CardIndex is int card && card >= 0)
        {
            return (MarkKind.Card, card);
        }
        if (o.Index is int index && index >= 0)
        {
            return (MarkKind.ScreenOption, index);
        }
        return null;
    }

    private static (MarkKind Kind, int Key)? Target(MarkPlanStep s) =>
        Target(new MarkOption(s.Label, s.Action, s.CardIndex, s.TargetIndex, s.AllyIndex, s.Slot,
            null, false, false, null, null, null));

    /// <summary>
    /// The marks for one advice: at most one per thing, best first, then the
    /// plan in order, then the alternatives by share.
    /// </summary>
    public static List<Mark> Build(IReadOnlyList<MarkOption> options, IReadOnlyList<MarkPlanStep>? plan)
    {
        double? bestQ = options.FirstOrDefault(o => o.IsBest)?.Q;

        // The plan's order, per thing; the first step is the pick whether or
        // not the advisor flagged it so.
        var order = new Dictionary<(MarkKind, int), (int Order, MarkPlanStep Step)>();
        if (plan != null)
        {
            for (int i = 0; i < plan.Count; i++)
            {
                if (Target(plan[i]) is { } t && !order.ContainsKey(t))
                {
                    order[t] = (i + 1, plan[i]);
                }
            }
        }

        // One option per thing: the best-ranked one (a targeted card has one
        // option per target).
        var byThing = new Dictionary<(MarkKind, int), MarkOption>();
        foreach (MarkOption o in options)
        {
            if (Target(o) is not { } t)
            {
                continue;
            }
            if (!byThing.TryGetValue(t, out MarkOption? have) || Better(o, have))
            {
                byThing[t] = o;
            }
        }

        var marks = new List<Mark>();
        foreach (((MarkKind kind, int key), MarkOption o) in byThing)
        {
            bool planned = order.TryGetValue((kind, key), out var step);
            MarkTier tier;
            if (o.IsBest || (planned && step.Order == 1))
            {
                tier = MarkTier.Best;
            }
            else if (planned || o.InBand || IsAsGood(o, bestQ))
            {
                tier = MarkTier.Good;
            }
            else if (Hidden(o))
            {
                continue;
            }
            else if ((o.Share ?? 0) >= AltShare)
            {
                tier = MarkTier.Alt;
            }
            else
            {
                continue;
            }
            int? markOrder = planned ? step.Order : tier == MarkTier.Best ? 1 : null;
            int? target = planned ? step.Step.TargetIndex ?? o.TargetIndex : o.TargetIndex;
            int? ally = planned ? step.Step.AllyIndex ?? o.AllyIndex : o.AllyIndex;
            marks.Add(new Mark(kind, key, tier, markOrder, target, ally, o.Label, o.Share, o.Q));
        }

        // A planned step the options did not list (an advisor that sends the
        // plan but trims options) still gets its number.
        foreach (((MarkKind kind, int key), (int ord, MarkPlanStep s)) in order)
        {
            if (!byThing.ContainsKey((kind, key)))
            {
                marks.Add(new Mark(kind, key, ord == 1 ? MarkTier.Best : MarkTier.Good, ord,
                    s.TargetIndex, s.AllyIndex, s.Label, null, null));
            }
        }

        marks.Sort((a, b) =>
        {
            int t = b.Tier.CompareTo(a.Tier);
            if (t != 0)
            {
                return t;
            }
            int oa = a.Order ?? int.MaxValue, ob = b.Order ?? int.MaxValue;
            if (oa != ob)
            {
                return oa.CompareTo(ob);
            }
            return (b.Share ?? 0).CompareTo(a.Share ?? 0);
        });

        // Keep the alternatives few: rings everywhere say nothing.
        int alts = 0;
        marks.RemoveAll(m => m.Tier == MarkTier.Alt && ++alts > MaxAlts);
        return marks;
    }

    private static bool Better(MarkOption a, MarkOption b)
    {
        if (a.IsBest != b.IsBest)
        {
            return a.IsBest;
        }
        if (a.PickRank is int ra && b.PickRank is int rb && ra != rb)
        {
            return ra < rb;
        }
        if (a.Q is double qa && b.Q is double qb && Math.Abs(qa - qb) > 1e-9)
        {
            return qa > qb;
        }
        return (a.Share ?? 0) > (b.Share ?? 0);
    }

    private static bool IsAsGood(MarkOption o, double? bestQ) =>
        o.Q is double q && bestQ is double b && q >= b - GoodQDelta && !Hidden(o);

    /// <summary>A move nobody would make: no mark, and folded away in the list.</summary>
    public static bool Hidden(MarkOption o) =>
        !o.IsBest && o.Share is double s && s < HiddenShare;

    // ------------------------------------------------------------------
    // Words for a row
    // ------------------------------------------------------------------

    /// <summary>
    /// A player's word for how an option compares with the pick. With a Q
    /// (the search) it is the loss against the best move; with only a
    /// probability (Instinct) it is how sure the net is.
    /// </summary>
    public static string Verdict(MarkOption o, double? bestQ)
    {
        if (o.IsBest)
        {
            return "Best";
        }
        if (o.InBand)
        {
            return "Also good";
        }
        if (o.Q is double q && bestQ is double b)
        {
            double loss = b - q;
            return loss <= GoodQDelta ? "Also good"
                : loss <= 0.02 ? "Okay"
                : loss <= 0.06 ? "Weak"
                : "Mistake";
        }
        if (o.Share is double s)
        {
            return s >= 0.25 ? "Also good" : s >= 0.08 ? "Okay" : "Unlikely";
        }
        return "";
    }

    /// <summary>
    /// The word for a ranked row that may be a later step of the turn's plan.
    /// A later step is judged at THIS decision, where it is not the move to
    /// make yet -- Instinct gave plan step 2 an 8% share and the row read
    /// "Unlikely" under a plan that said to play it next (live, 2026-11-07).
    /// It reads "Then" instead.
    /// </summary>
    public static string RowVerdict(MarkOption o, double? bestQ, int? planStep) =>
        planStep is int step && step >= 2 && !o.IsBest ? "Then" : Verdict(o, bestQ);

    // ------------------------------------------------------------------
    // No raw ids on screen
    // ------------------------------------------------------------------

    // An id the game writes in its tables: SHRUG_IT_OFF, CARD.SHRUG_IT_OFF,
    // STRIKE_IRONCLAD. Two or more upper-case words joined by underscores, or
    // a dotted CATEGORY.ID. A lone upper-case word ("AI", "HP") is left alone.
    private static readonly Regex IdToken = new(
        @"\b(?:[A-Z][A-Z0-9]*\.)?[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b|\b[A-Z][A-Z0-9]*\.[A-Z][A-Z0-9_]*\b",
        RegexOptions.CultureInvariant);

    /// <summary>
    /// The last guard before a label reaches the screen: any raw game id in it
    /// becomes words ("SHRUG_IT_OFF" -> "Shrug It Off"). The advisor sends
    /// display names; this keeps a surface it missed from leaking an id. A
    /// <paramref name="lookup"/> (the game's own titles) wins over the guess.
    /// </summary>
    public static string Humanize(string? label, Func<string, string?>? lookup = null)
    {
        if (string.IsNullOrEmpty(label))
        {
            return label ?? "";
        }
        return IdToken.Replace(label, m =>
        {
            string id = m.Value;
            int dot = id.LastIndexOf('.');
            string bare = dot >= 0 ? id[(dot + 1)..] : id;
            string? known = null;
            try
            {
                known = lookup?.Invoke(bare);
            }
            catch (Exception)
            {
                known = null;
            }
            return string.IsNullOrWhiteSpace(known) ? TitleCase(bare) : known!;
        });
    }

    private static readonly HashSet<string> SmallWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to",
    };

    /// <summary>"SHRUG_IT_OFF" -> "Shrug It Off" (small words lower after the first).</summary>
    public static string TitleCase(string id)
    {
        string[] words = id.Split('_', StringSplitOptions.RemoveEmptyEntries);
        var sb = new StringBuilder();
        for (int i = 0; i < words.Length; i++)
        {
            string w = words[i].ToLower(CultureInfo.InvariantCulture);
            if (i > 0)
            {
                sb.Append(' ');
            }
            if (i > 0 && SmallWords.Contains(w))
            {
                sb.Append(w);
                continue;
            }
            sb.Append(char.ToUpperInvariant(w[0]));
            sb.Append(w, 1, w.Length - 1);
        }
        return sb.ToString();
    }
}
