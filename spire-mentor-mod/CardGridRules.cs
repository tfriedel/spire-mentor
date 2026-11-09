// CardGridRules.cs -- what a card grid screen (NCardGridSelectionScreen)
// offers once some cards are already selected.
//
// Game-free on purpose (no Godot, no sts2.dll) so tests/unit can link it.

namespace SpireMentor;

internal static class CardGridRules
{
    /// <summary>The combat pile picker (Neow's Fury, Headbutt, ...).</summary>
    public const string CombatPileScreen = "NCombatPileCardSelectScreen";

    /// <summary>
    /// Whether a grid with <paramref name="selected"/> cards picked keeps
    /// offering the remaining <paramref name="remaining"/> cards, instead of
    /// asking only to confirm.
    ///
    /// <para>Most grids only wait for a confirm once a card is selected.
    /// Offering their cards again let autoplay pick a 2nd, 3rd, ... card
    /// (NDeckEnchantSelectScreen, 2026-08-06). A combat pile pick of "up to
    /// N" is different: the player may still take more. Offering only
    /// confirm there cut every Neow's Fury (up to 2 from the discard) to one
    /// card (issue #1206). The advisor answers `skip` (confirm) itself once
    /// its picks are done (`selected_count` in the state).</para>
    ///
    /// <para>With no card left to pick, it is confirm-only again: offering
    /// nothing would leave the screen without a decision.</para>
    /// </summary>
    public static bool OffersMorePicks(string screen, int selected, int maxSelect, int remaining) =>
        selected <= 0 || (screen == CombatPileScreen && selected < maxSelect && remaining > 0);
}
