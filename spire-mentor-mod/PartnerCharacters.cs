// PartnerCharacters.cs -- which character each AI partner seat plays, and
// how long a partner holds a move for a teammate's setup play.
//
// Characters: clicking an AI partner's slot in the lobby cycles its
// character (SimAiPartnerLobby; right-click goes back), and the pick is kept
// per seat in SpireMentorModConfig.AiPartnerCharacters ("SILENT,,DEFECT":
// seat 1 Silent, seat 2 unset, seat 3 Defect), so the next lobby starts from
// it. An unset seat plays the character after Ironclad in seat order (seat 1
// the second character, ...), as partners always have.
//
// Holds: SimAiPartnerBrain holds a move whose advice says a teammate's setup
// play would improve it, for a human teammate up to the player's setting, for
// another AI partner only briefly (it plays within a second or two; two
// partners holding for each other would both sit out a human-sized hold).
//
// Game-free on purpose (no Godot, no sts2.dll) so tests/unit can link it.

using System;
using System.Collections.Generic;
using System.Linq;

namespace SpireMentor;

internal static class PartnerCharacters
{
    /// <summary>The settings slider's range for the human setup hold.</summary>
    public const double MaxHumanHoldSeconds = 60.0;

    /// <summary>The hold when only AI partners are still to act.</summary>
    public const double AiHoldSeconds = 2.5;

    /// <summary>The index <paramref name="direction"/> steps (+1 next, -1
    /// back) from <paramref name="current"/> among <paramref name="count"/>
    /// characters, wrapping. An unknown current (-1) starts at the first
    /// (next) or the last (back).</summary>
    public static int Cycle(int current, int count, int direction)
    {
        if (count <= 0)
        {
            return -1;
        }
        if (current < 0 || current >= count)
        {
            return direction < 0 ? count - 1 : 0;
        }
        int step = Math.Sign(direction);
        return ((current + step) % count + count) % count;
    }

    /// <summary>The seat's default character index when nothing is saved:
    /// the one after the first, in seat order.</summary>
    public static int DefaultIndex(int seat, int count) => count <= 0 ? -1 : (seat + 1) % count;

    /// <summary>The character id saved for <paramref name="seat"/> (0-based)
    /// in <paramref name="prefs"/>, or null.</summary>
    public static string? ForSeat(string? prefs, int seat)
    {
        string[] ids = Split(prefs);
        return seat >= 0 && seat < ids.Length && ids[seat].Length > 0 ? ids[seat] : null;
    }

    /// <summary><paramref name="prefs"/> with <paramref name="seat"/> set to
    /// <paramref name="id"/>; trailing unset seats are dropped.</summary>
    public static string WithSeat(string? prefs, int seat, string id)
    {
        if (seat < 0)
        {
            return prefs ?? "";
        }
        List<string> ids = Split(prefs).ToList();
        while (ids.Count <= seat)
        {
            ids.Add("");
        }
        ids[seat] = id.Replace(",", "").Trim();
        while (ids.Count > 0 && ids[^1].Length == 0)
        {
            ids.RemoveAt(ids.Count - 1);
        }
        return string.Join(",", ids);
    }

    /// <summary>The longest a partner holds a move for a teammate's setup
    /// play: the player's setting (clamped to 0-60 s) while a human is still
    /// to act, else the short AI hold; 0 (never) wins either way.</summary>
    public static double HoldSeconds(double setting, bool humanStillToAct)
    {
        double human = double.IsNaN(setting) ? 0.0 : Math.Clamp(setting, 0.0, MaxHumanHoldSeconds);
        if (human <= 0.0)
        {
            return 0.0;
        }
        return humanStillToAct ? human : Math.Min(human, AiHoldSeconds);
    }

    private static string[] Split(string? prefs) =>
        (prefs ?? "").Split(',').Select(s => s.Trim()).ToArray();
}
