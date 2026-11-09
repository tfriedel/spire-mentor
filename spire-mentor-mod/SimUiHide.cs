// SimUiHide.cs -- "hide everything for a quick look" (Ctrl+H / the pill).
//
// One toggle that clears the screen of the mod without rearranging it: every
// SpireMentor*Layer except the toolbar goes invisible, the toolbar shrinks to
// its pill (click it, or Ctrl+H again, to bring everything back), and showing
// restores exactly the layers that were visible before -- an open log stays
// open, a closed help panel stays closed. Dragging panels out of the way is
// the slow alternative this replaces.
//
// The layers whose visibility is recomputed on every refresh (HUD, coach
// chip, coach panel) also AND in !Hidden, so new advice arriving while hidden
// cannot pop them back up. Session-only, like the toolbar's dragged position.
// Main thread only.

using System;
using System.Collections.Generic;
using Godot;

namespace SpireMentor;

internal static class SimUiHide
{
    private const string ToolbarLayerName = "SpireMentorToolbarLayer";

    /// <summary>The mod's UI is hidden down to the toolbar pill.</summary>
    public static bool Hidden { get; private set; }

    private static readonly List<CanvasLayer> HiddenByUs = new();

    public static void Toggle()
    {
        try
        {
            if (Engine.GetMainLoop() is not SceneTree tree)
            {
                return;
            }
            Hidden = !Hidden;
            if (Hidden)
            {
                HiddenByUs.Clear();
                foreach (Node child in tree.Root.GetChildren())
                {
                    if (child is CanvasLayer layer && layer.Visible &&
                        layer.Name.ToString().StartsWith("SpireMentor", StringComparison.Ordinal) &&
                        layer.Name != ToolbarLayerName)
                    {
                        layer.Visible = false;
                        HiddenByUs.Add(layer);
                    }
                }
            }
            else
            {
                foreach (CanvasLayer layer in HiddenByUs)
                {
                    if (GodotObject.IsInstanceValid(layer))
                    {
                        layer.Visible = true;
                    }
                }
                HiddenByUs.Clear();
            }
            GD.Print($"[SpireMentor] UI {(Hidden ? "hidden (Ctrl+H or the Spire Mentor badge shows it)" : "shown")}");
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimUiHide] Toggle failed: {ex.Message}");
        }
        // The self-refreshing layers (HUD, chip, panel) and the toolbar
        // re-evaluate against Hidden; this also corrects a restored layer
        // whose own condition changed while it was hidden.
        SimRuntime.NotifyChanged();
    }
}
