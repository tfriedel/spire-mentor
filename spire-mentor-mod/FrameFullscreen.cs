// FrameFullscreen.cs -- the game's own Fullscreen switch while it sits in the
// Spire Mentor frame.
//
// Framed, "fullscreen" is the frame's: the panels move into the game
// (FrameLink.UiOutside off) and the frame's page has its window go borderless
// over the whole monitor, so the game fills the screen. The game's own
// fullscreen cannot do that: its window is the frame's WS_CHILD, and
// switching it to Fullscreen only fights the embedding (the player saw no
// change). So while framed, the game's switch -- the settings tickbox and
// Alt+Enter, which both end in NFullscreenTickbox.SetFullscreen -- drives the
// frame's fullscreen instead of the window mode, and the saved setting
// follows the frame (Sync), so the tickbox shows what the player sees and
// Alt+Enter toggles both ways. And the game applies its display settings
// windowed while framed (FrameWindowedDisplayPatch), whatever is saved.
//
// Unframed, nothing changes.

using System;
using HarmonyLib;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Screens.Settings;
using MegaCrit.Sts2.Core.Saves;

namespace SpireMentor;

[HarmonyPatch(typeof(NFullscreenTickbox), nameof(NFullscreenTickbox.SetFullscreen))]
public static class FrameFullscreenPatch
{
    [HarmonyPrefix]
    static bool Prefix(bool fullscreen)
    {
        if (!FrameLink.Framed)
        {
            return true;
        }
        try
        {
            Logger.Log($"[FrameFullscreen] the game's Fullscreen switch ({(fullscreen ? "on" : "off")}): the frame's fullscreen instead");
            FrameLink.SetUiOutside(!fullscreen);
            FrameFullscreen.Sync();
        }
        catch (Exception ex)
        {
            Logger.Log($"[FrameFullscreen] ignoring error (never breaks gameplay): {ex.Message}");
        }
        return false;
    }
}

/// <summary>
/// Framed, the game's window always applies its display settings windowed:
/// a saved Fullscreen setting would put the frame's child window into
/// fullscreen and fight the embedding -- at startup in a game the frame
/// launched (the dev lane: the mod is framed before the game first applies
/// its settings) and on any later apply from the settings screen. The saved
/// value is lowered for the call only and put back (it says whether the
/// FRAME is fullscreen; FrameFullscreen.Sync). Both supported builds read it
/// there (v0.111.0 through ResolveFullscreen, v0.107.1 directly).
/// </summary>
[HarmonyPatch(typeof(NGame), nameof(NGame.ApplyDisplaySettings))]
public static class FrameWindowedDisplayPatch
{
    [HarmonyPrefix]
    static void Prefix(out bool __state)
    {
        __state = false;
        if (!FrameLink.Framed)
        {
            return;
        }
        try
        {
            SettingsSave? save = SaveManager.Instance?.SettingsSave;
            if (save is { Fullscreen: true })
            {
                Logger.Log("[FrameFullscreen] framed: applying the display settings windowed (the frame does fullscreen)");
                save.Fullscreen = false;
                __state = true;
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[FrameFullscreen] ignoring error (never breaks gameplay): {ex.Message}");
        }
    }

    [HarmonyPostfix]
    static void Postfix(bool __state)
    {
        if (!__state)
        {
            return;
        }
        try
        {
            if (SaveManager.Instance?.SettingsSave is { } save)
            {
                save.Fullscreen = true;
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[FrameFullscreen] ignoring error (never breaks gameplay): {ex.Message}");
        }
    }
}

internal static class FrameFullscreen
{
    /// <summary>The frame the mod started has closed and handed the game
    /// its window back. Closed in fullscreen (the saved setting says so,
    /// Sync), the game goes fullscreen on its own, rather than leaving the
    /// player in a window with the setting claiming fullscreen. Main thread,
    /// after FrameLink.Detach.</summary>
    public static void AfterFrameClosed()
    {
        if (FrameLink.Framed)
        {
            return;
        }
        try
        {
            if (SaveManager.Instance?.SettingsSave is { Fullscreen: true })
            {
                Logger.Log("[FrameFullscreen] the frame closed in fullscreen: the game goes fullscreen itself");
                NFullscreenTickbox.SetFullscreen(true);
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[FrameFullscreen] ignoring error (never breaks gameplay): {ex.Message}");
        }
    }

    /// <summary>Framed: the game's saved Fullscreen setting says whether the
    /// frame is fullscreen (the panels in the game). Main thread.</summary>
    public static void Sync()
    {
        if (!FrameLink.Framed)
        {
            return;
        }
        try
        {
            SettingsSave? save = SaveManager.Instance?.SettingsSave;
            if (save != null)
            {
                save.Fullscreen = !FrameLink.UiOutside;
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[FrameFullscreen] could not sync the Fullscreen setting: {ex.Message}");
        }
    }
}
