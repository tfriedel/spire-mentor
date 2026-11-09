// AdvisorBrainRules.cs -- which brain answers the player's advice: Search or Instinct.
//
// Search is what the advisor always was: MCTS for the fights, the PPO run
// policy for the screens. Instinct is the end-to-end net (no search) playing
// every decision, today the e2e Ironclad net on v0.111.0 only. The player
// picks one (SpireMentorModConfig.AdvisorBrain, the badge's Brain switch,
// Shift+F8); the mod sends the choice on every advice_request as `brain`, and
// the ADVISOR decides per request whether Instinct can answer (PROTOCOL.md
// section 5, v5.2). The mod never filters by character itself: it only uses
// the advisor's `capabilities.instinct` to decide whether the switch is worth
// showing.
//
// Game-free (System.Text.Json only) so tests/unit pins the rules: the
// capability parse, who sees the switch, which reply is stale after a
// toggle, and the HUD's source tag.

using System;
using System.Collections.Generic;
using System.Text.Json;

namespace SpireMentor;

/// <summary>One `capabilities.instinct` entry: a character + game build this
/// advisor's Instinct can play.</summary>
internal sealed record InstinctTarget(string Character, string GameVersion);

internal static class AdvisorBrainRules
{
    public const string Search = "search";
    public const string Instinct = "instinct";

    /// <summary>The wire value for a stored preference: anything but
    /// "instinct" (case-insensitive, trimmed) is Search, so a blank or
    /// hand-edited config can never send a brain the advisor does not know.</summary>
    public static string Normalize(string? value) =>
        string.Equals(value?.Trim(), Instinct, StringComparison.OrdinalIgnoreCase) ? Instinct : Search;

    public static bool IsInstinct(string? value) => Normalize(value) == Instinct;

    /// <summary>
    /// SpireMentor.conf's <c>advisor_brain</c>: "search" / "instinct"
    /// (case-insensitive, trimmed), else null -- absent, blank or a typo.
    /// Unlike <see cref="Normalize"/> a typo is NOT read as Search: the key
    /// overrides the player's setting, and an override nobody meant must not
    /// happen (SimConfig logs it instead).
    /// </summary>
    public static string? ParseConf(string? value)
    {
        string v = value?.Trim() ?? "";
        if (v.Equals(Instinct, StringComparison.OrdinalIgnoreCase))
        {
            return Instinct;
        }
        return v.Equals(Search, StringComparison.OrdinalIgnoreCase) ? Search : null;
    }

    /// <summary>The brain in effect: the session's conf override while it
    /// stands (null once the player switched in game), else the player's
    /// persisted preference.</summary>
    public static string Effective(string? sessionOverride, string? preference) =>
        Normalize(ParseConf(sessionOverride) ?? preference);

    public static string Toggle(string? current) => IsInstinct(current) ? Search : Instinct;

    /// <summary>The player-facing name ("Search" / "Instinct").</summary>
    public static string Label(string? value) => IsInstinct(value) ? "Instinct" : "Search";

    /// <summary>
    /// `capabilities.instinct` of an advice reply: top level, next to
    /// `advice`, as the advisor sends it; `advice.capabilities` is accepted
    /// too (the same two places as `capabilities.coach`). Empty when absent,
    /// malformed, or the advisor has no Instinct net -- an older advisor.
    /// Entries without both fields are skipped.
    /// </summary>
    public static IReadOnlyList<InstinctTarget> ParseCapability(JsonElement root)
    {
        var targets = new List<InstinctTarget>();
        if (root.ValueKind != JsonValueKind.Object)
        {
            return targets;
        }
        if (!TryInstinctArray(root, out JsonElement arr) &&
            !(root.TryGetProperty("advice", out JsonElement advice) && TryInstinctArray(advice, out arr)))
        {
            return targets;
        }
        foreach (JsonElement e in arr.EnumerateArray())
        {
            if (e.ValueKind != JsonValueKind.Object)
            {
                continue;
            }
            string? character = ReadString(e, "character");
            string? version = ReadString(e, "game_version");
            if (!string.IsNullOrWhiteSpace(character) && !string.IsNullOrWhiteSpace(version))
            {
                targets.Add(new InstinctTarget(character.Trim(), version.Trim()));
            }
        }
        return targets;
    }

    private static bool TryInstinctArray(JsonElement owner, out JsonElement arr)
    {
        arr = default;
        return owner.ValueKind == JsonValueKind.Object &&
               owner.TryGetProperty("capabilities", out JsonElement caps) &&
               caps.ValueKind == JsonValueKind.Object &&
               caps.TryGetProperty("instinct", out arr) &&
               arr.ValueKind == JsonValueKind.Array;
    }

    /// <summary>
    /// The advisor's Instinct can play this run: its capability names the
    /// run's character and game build. Case-insensitive on both (the run
    /// says "IRONCLAD", the advisor "Ironclad"); the build compares in wire
    /// form ("v0.111.0", GameVersionFormat) so "0.111.0" matches too.
    /// </summary>
    public static bool Eligible(IReadOnlyList<InstinctTarget> targets, string? character, string? gameVersion)
    {
        if (targets.Count == 0 || string.IsNullOrWhiteSpace(character))
        {
            return false;
        }
        string build = GameVersionFormat.Normalize(gameVersion) ?? GameVersionFormat.Default;
        foreach (InstinctTarget t in targets)
        {
            if (string.Equals(t.Character, character.Trim(), StringComparison.OrdinalIgnoreCase) &&
                string.Equals(GameVersionFormat.Normalize(t.GameVersion), build, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// The brain to ask the advisor for: the player's preference, except that
    /// Instinct is asked for only in a run it plays. Elsewhere the search
    /// answers without the "Search (Instinct: ...)" tag on every reply:
    /// Instinct is the default, and most runs are not ones it plays.
    /// <paramref name="known"/>: the advisor has said which runs Instinct
    /// plays (its capability); until then the preference goes as it is and
    /// the advisor falls back where it must. The toolbar's switch is shown
    /// exactly where this can be Instinct (<paramref name="eligible"/>).
    /// </summary>
    public static string Requested(string? preference, bool known, bool eligible) =>
        IsInstinct(preference) && (eligible || !known) ? Instinct : Search;

    /// <summary>
    /// A reply that must not be rendered because the player switched brain
    /// after it was asked for: <paramref name="requested"/> is the brain the
    /// request was stamped with, <paramref name="current"/> the preference now.
    ///
    /// Deliberately NOT keyed on the reply's own `advice.brain`: a Search
    /// answer to an Instinct request is the advisor falling back
    /// (`instinct_unavailable`), the right thing to show; and an Instinct
    /// answer to a Search request can only be an advisor bug, where dropping
    /// it would leave the decision with no advice at all. The stamp is what
    /// tells a reply from before the toggle, and it cannot loop: the
    /// re-request is stamped with the new preference.
    /// </summary>
    public static bool IsStale(string requested, string current) => Normalize(requested) != Normalize(current);

    /// <summary>
    /// The HUD's source tag for a reply: "Instinct" when the net answered,
    /// "Search (Instinct: reason)" when Instinct was asked for and the search
    /// answered instead, null for a plain search reply (the common case,
    /// which says nothing).
    /// </summary>
    public static string? SourceTag(string? answered, string? instinctUnavailable)
    {
        if (answered != null && IsInstinct(answered))
        {
            return "Instinct";
        }
        if (!string.IsNullOrWhiteSpace(instinctUnavailable))
        {
            return $"Search (Instinct: {instinctUnavailable.Trim()})";
        }
        return null;
    }

    /// <summary>
    /// The advice note as the toolbar's status line shows it, with the source
    /// in front when it is not the plain search: "Instinct · note", or the
    /// fallback tag when the advisor's note does not already carry the reason
    /// (the advisor puts it there too; it must not read twice).
    /// </summary>
    public static string? NoteWithSource(string? note, string? answered, string? instinctUnavailable)
    {
        string? tag = SourceTag(answered, instinctUnavailable);
        if (tag == null)
        {
            return note;
        }
        if (string.IsNullOrWhiteSpace(note))
        {
            return tag;
        }
        if (!IsInstinct(answered) && note.Contains(instinctUnavailable!.Trim(), StringComparison.OrdinalIgnoreCase))
        {
            return note;
        }
        return $"{tag} · {note}";
    }

    private static string? ReadString(JsonElement el, string name) =>
        el.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
}
