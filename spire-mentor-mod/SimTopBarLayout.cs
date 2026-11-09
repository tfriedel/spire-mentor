// SimTopBarLayout.cs -- shared "stay clear of the game's top bar" probe.
//
// Extracted from SimHud.RepositionPanels/TopBar so the toolbar and the HUD
// agree on where the safe top edge is: SafeTopY on the left (the advice
// column), SafeTopYRight on the right (the Spire Mentor badge). Measured from the portrait/HP/gold
// widgets, NOT from NTopBar itself: the bar node is anchored full-rect (its
// own GetGlobalRect is the whole viewport), so clearing *it* threw the chip
// off the bottom of the screen. Only ever pushes DOWN: on a screen with no
// run in progress there is no bar and callers keep their corner.

using System;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Relics;

namespace SpireMentor;

internal static class SimTopBarLayout
{
    /// <summary>Y when no top bar is on screen (main menu, character select).</summary>
    public const float FallbackTop = 16f;

    /// <summary>Breathing room between the game's top bar and our panels.</summary>
    public const float TopBarClearance = 12f;

    private static NTopBar? _topBar;
    private static int _topBarSearchCooldown;

    /// <summary>
    /// The lowest safe Y below the top-bar widgets, clamped to the top
    /// quarter of the screen (whatever the layout does, panels stay in the
    /// top quarter rather than sliding off it). <paramref name="viewportSource"/>
    /// is any control in the tree, used only to read the viewport size.
    /// </summary>
    public static float SafeTopY(Control viewportSource)
    {
        float top = FallbackTop;
        try
        {
            NTopBar? bar = TopBar();
            if (bar != null && bar.IsVisibleInTree())
            {
                foreach (Control? widget in new Control?[] { bar.Portrait, bar.Hp, bar.Gold })
                {
                    if (widget == null || !GodotObject.IsInstanceValid(widget) ||
                        !widget.IsVisibleInTree())
                    {
                        continue;
                    }
                    Rect2 rect = widget.GetGlobalRect();
                    top = Math.Max(top, rect.Position.Y + rect.Size.Y + TopBarClearance);
                }
                // The relic tray hangs BELOW the top bar on the left -- exactly
                // where "under the bar widgets" is. Clear its lowest holder too
                // (it wraps to more rows as relics accumulate), same idiom the
                // game itself uses (NTopBar.cs: GlobalUi.RelicInventory.RelicNodes).
                NRelicInventory? relics = NRun.Instance?.GlobalUi?.RelicInventory;
                if (relics != null && GodotObject.IsInstanceValid(relics) &&
                    relics.IsVisibleInTree())
                {
                    foreach (NRelicInventoryHolder holder in relics.RelicNodes)
                    {
                        if (holder == null || !GodotObject.IsInstanceValid(holder) ||
                            !((Control)holder).IsVisibleInTree())
                        {
                            continue;
                        }
                        Rect2 rect = ((Control)holder).GetGlobalRect();
                        top = Math.Max(top, rect.Position.Y + rect.Size.Y + TopBarClearance);
                    }
                }
                // Relic rows can legitimately push past the old quarter-screen
                // clamp late in a run; allow up to a third before giving up.
                float ceiling = viewportSource.GetViewportRect().Size.Y * 0.33f;
                top = Math.Min(top, Math.Max(FallbackTop, ceiling));
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimTopBarLayout] SafeTopY failed: {ex.Message}");
        }
        return top;
    }

    /// <summary>
    /// The right edge of the co-op player list when it is on screen
    /// (<c>NGlobalUi.MultiplayerPlayerContainer</c>: one row per teammate --
    /// name, HP bar, block, powers -- under the relic tray on the left), else
    /// null (single player, or the player hid the bars). The docked advice
    /// column moves right of it: below it would sit on the characters once
    /// three or four players are listed, and on it (as through beta.6) hid the
    /// teammates' HP and powers behind the advice.
    /// </summary>
    public static float? CoopRosterRight()
    {
        try
        {
            Control? roster = NRun.Instance?.GlobalUi?.MultiplayerPlayerContainer;
            if (roster == null || !GodotObject.IsInstanceValid(roster) ||
                !roster.IsVisibleInTree() || roster.GetChildCount() == 0)
            {
                return null;
            }
            float right = float.NegativeInfinity;
            foreach (Node child in roster.GetChildren())
            {
                if (child is Control row && GodotObject.IsInstanceValid(row) && row.IsVisibleInTree())
                {
                    Rect2 rect = row.GetGlobalRect();
                    right = Math.Max(right, rect.Position.X + rect.Size.X);
                }
            }
            return right > 0f ? right : null;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimTopBarLayout] CoopRosterRight failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// The lowest safe Y below the top bar's RIGHT-hand widgets -- the map,
    /// deck and settings buttons and the run timer -- where the Spire Mentor
    /// badge docks. The right side has no relic tray and no co-op player list
    /// hanging under it (both sit on the left, NMultiplayerPlayerStateContainer
    /// parks under the relics), so this is usually just under the bar. Same
    /// rules as <see cref="SafeTopY"/>: only pushes down, clamped to the top
    /// third.
    /// </summary>
    public static float SafeTopYRight(Control viewportSource)
    {
        float top = FallbackTop;
        try
        {
            NTopBar? bar = TopBar();
            if (bar != null && bar.IsVisibleInTree())
            {
                foreach (Control? widget in new Control?[] { bar.Map, bar.Deck, bar.Pause, bar.Timer })
                {
                    if (widget == null || !GodotObject.IsInstanceValid(widget) ||
                        !widget.IsVisibleInTree())
                    {
                        continue;
                    }
                    Rect2 rect = widget.GetGlobalRect();
                    top = Math.Max(top, rect.Position.Y + rect.Size.Y + TopBarClearance);
                }
                float ceiling = viewportSource.GetViewportRect().Size.Y * 0.33f;
                top = Math.Min(top, Math.Max(FallbackTop, ceiling));
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimTopBarLayout] SafeTopYRight failed: {ex.Message}");
        }
        return top;
    }

    /// <summary>
    /// The run's top bar, cached. There is no NTopBar.Instance, so this
    /// falls back to a whole-tree walk -- throttled to once a second, because
    /// outside a run the walk would otherwise run 20x/s and never find one.
    /// </summary>
    public static NTopBar? TopBar()
    {
        if (_topBar != null && GodotObject.IsInstanceValid(_topBar) && _topBar.IsInsideTree())
        {
            return _topBar;
        }
        if (--_topBarSearchCooldown > 0)
        {
            return null;
        }
        _topBarSearchCooldown = 20;
        _topBar = Engine.GetMainLoop() is SceneTree tree
            ? UiHelper.FindFirst<NTopBar>(tree.Root)
            : null;
        return _topBar;
    }
}
