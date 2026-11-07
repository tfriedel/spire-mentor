// PotionSwapAdvice.cs -- the full-belt potion swap as an advice lane (v5.2).
//
// The autopilot has offered a potion reward that only a full belt blocks as a
// `potion_swap` option group on `reward_screen` since GH #1069: decline first,
// then "discard the potion in slot k and take this one" per belt slot. The
// ADVISOR lane never asked it: the loot screen is answered mod-side ("claim
// everything, then proceed", SimScreenAdvice.DescribeRewards), and a blocked
// potion is not claimable, so the player was simply walked past it.
//
// Now, once everything claimable is claimed and such a potion is still on
// the screen, the mod asks `advice_for: "potion_swap"` with that one group as
// its options -- but only of an advisor that advertises
// `capabilities.potion_swap >= 1`. Without it nothing changes. Autobattle
// stops on it like any run-level decision (Autobattle.IsStop: not local, not
// in combat): giving up a held potion is a real choice, not a loot click.
//
// Game-free (System.Text.Json only) so tests/unit pins the rules; the screen
// side (reading the belt, discarding, clicking) is SimScreenAdvice.

using System;
using System.Collections.Generic;
using System.Text.Json;

namespace SpireMentor;

internal static class PotionSwapAdvice
{
    /// <summary>The `advice_for` / `human_action.decision_type` value, and
    /// the option `action` (NonCombatBridgeProtocol.PotionSwapAction).</summary>
    public const string AdviceFor = "potion_swap";

    /// <summary>Label of the "leave it on the screen" option, index 0.</summary>
    public const string DeclineLabel = "decline";

    /// <summary>
    /// `capabilities.potion_swap` of an advice reply: top level, next to
    /// `advice` (where the advisor sends it), else `advice.capabilities`; 0
    /// when absent or not a non-negative integer -- an older advisor.
    /// </summary>
    public static int CapabilityOf(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
        {
            return 0;
        }
        if (TryLevel(root, out int level))
        {
            return level;
        }
        return root.TryGetProperty("advice", out JsonElement advice) && TryLevel(advice, out level) ? level : 0;
    }

    private static bool TryLevel(JsonElement owner, out int level)
    {
        level = 0;
        if (owner.ValueKind == JsonValueKind.Object &&
            owner.TryGetProperty("capabilities", out JsonElement caps) &&
            caps.ValueKind == JsonValueKind.Object &&
            caps.TryGetProperty(AdviceFor, out JsonElement v) &&
            v.ValueKind == JsonValueKind.Number &&
            v.TryGetInt32(out int n))
        {
            level = Math.Max(0, n);
            return true;
        }
        return false;
    }

    /// <summary>
    /// Ask the advisor about a blocked potion: nothing claimable is left on
    /// the screen (the mod claims those first, as the autopilot's layout puts
    /// them first), at least one full-belt potion reward has not been
    /// declined, and the advisor understands the question.
    /// </summary>
    public static bool ShouldAsk(int claimableCount, int blockedCount, int capability) =>
        claimableCount == 0 && blockedCount > 0 && capability >= 1;

    /// <summary>
    /// The option labels, in option index order: "decline", then
    /// "discard &lt;POTION_ID&gt;" per belt slot -- the autopilot's labels
    /// (SimRewardsScreenHandler.SwapOption), so both lanes spell the group
    /// the same way. An empty slot (cannot happen on a full belt) reads
    /// "discard empty", as there.
    /// </summary>
    public static List<string> Labels(IReadOnlyList<string?> belt)
    {
        var labels = new List<string>(belt.Count + 1) { DeclineLabel };
        foreach (string? id in belt)
        {
            labels.Add($"discard {id ?? "empty"}");
        }
        return labels;
    }

    /// <summary>
    /// The belt slot option <paramref name="index"/> discards, or null for
    /// the decline. Anything outside the group (a `skip`, which the takeover
    /// sends as index = option count) is a decline too: it never discards a
    /// potion nobody named.
    /// </summary>
    public static int? DiscardSlotOf(int index, int beltCount) =>
        index >= 1 && index <= beltCount ? index - 1 : null;
}
