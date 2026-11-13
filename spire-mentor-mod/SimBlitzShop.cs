// SimBlitzShop.cs -- the Blitz merchant's wares (v0.113.0+), read by reflection.
//
// Blitz (v0.113.0) replaces the shop's stock with "choices": five wares plus
// one cursed one (MerchantInventory.GetBlitzEntries), all free, of which the
// player takes up to `BlitzChoicesRemaining` (3). Each is a
// `BlitzMerchantEntry` whose purchase opens its own follow-up screen (a card
// reward, an upgrade/transform/enchant pick, a relic) that the existing
// describers already handle.
//
// One DLL binds against every supported build, and `BlitzMerchantEntry` does
// not exist before v0.113.0: naming the type would fail the JIT of whatever
// method mentions it on the older builds -- i.e. the whole shop. So the type
// and its members are looked up by name, the same way the v0.113.0
// CardSelectorStack bridge is (SimGameApi.InstallSelectorBridge), and every
// read degrades to "not a Blitz ware" when the build has none.

using System;
using System.Collections.Generic;
using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Localization;

namespace SpireMentor;

internal static class SimBlitzShop
{
    private static readonly Type? EntryType =
        AccessTools.TypeByName("MegaCrit.Sts2.Core.Entities.Merchant.BlitzMerchantEntry");

    private static readonly PropertyInfo? EntryKind = Prop(EntryType, "MerchantEntryType");
    private static readonly PropertyInfo? EntryCursed = Prop(EntryType, "IsCursed");
    private static readonly PropertyInfo? EntryRare = Prop(EntryType, "IsRare");
    private static readonly PropertyInfo? EntryTitle = Prop(EntryType, "Title");
    private static readonly PropertyInfo? EntryDescription = Prop(EntryType, "Description");
    private static readonly PropertyInfo? ChoicesLeft = Prop(typeof(MerchantInventory), "BlitzChoicesRemaining");
    private static readonly PropertyInfo? IsBlitzInventory = Prop(typeof(MerchantInventory), "IsBlitzShop");

    private static PropertyInfo? Prop(Type? type, string name) =>
        type == null ? null : AccessTools.Property(type, name);

    /// <summary>The entry is one of the Blitz merchant's choices.</summary>
    internal static bool IsBlitzEntry(MerchantEntry? entry) =>
        entry != null && EntryType != null && EntryType.IsInstanceOfType(entry);

    /// <summary>
    /// The ware's kind as the game names it (`BlitzMerchantEntryType`, e.g.
    /// `CardRewards`, `Relic`, `TwoRelics`), or null for a non-Blitz entry.
    /// </summary>
    internal static string? Kind(MerchantEntry? entry) =>
        IsBlitzEntry(entry) ? EntryKind?.GetValue(entry)?.ToString() : null;

    /// <summary>
    /// The wire `item` of a Blitz ware: its kind, whether it costs a curse
    /// (`IsCursed`: every kind from `Glam` on adds a random Curse) or is one
    /// of the rare kinds, and the game's own title/description text.
    /// </summary>
    internal static Dictionary<string, object?>? Detail(MerchantEntry? entry)
    {
        if (!IsBlitzEntry(entry))
            return null;
        return new Dictionary<string, object?>
        {
            ["blitz_kind"] = Kind(entry),
            ["cursed"] = EntryCursed?.GetValue(entry) as bool? ?? false,
            ["rare"] = EntryRare?.GetValue(entry) as bool? ?? false,
            ["title"] = Text(EntryTitle, entry),
            ["description"] = Text(EntryDescription, entry),
        };
    }

    /// <summary>
    /// How many more wares the Blitz merchant lets the player take, or null
    /// when the inventory is not a Blitz shop (or the build has none).
    /// </summary>
    internal static int? ChoicesRemaining(MerchantInventory? inventory)
    {
        if (inventory == null || IsBlitzInventory?.GetValue(inventory) is not true)
            return null;
        return ChoicesLeft?.GetValue(inventory) as int?;
    }

    private static string? Text(PropertyInfo? prop, MerchantEntry? entry)
    {
        try
        {
            return (prop?.GetValue(entry) as LocString)?.GetFormattedText();
        }
        catch (Exception)
        {
            // A title whose variables are not bound yet must not cost the
            // whole shop description.
            return null;
        }
    }
}
