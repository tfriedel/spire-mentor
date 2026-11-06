// SimAiPartnerMenu.cs -- the "With AI partners" entry in the game's
// multiplayer host menu, and the names the lobby shows for an AI-partner
// session (docs/plans/2026-11-05-ai-coop-partners.md, phase 1).
//
// The host menu (NMultiplayerHostSubmenu: Standard / Daily / Custom) is only
// reachable when no co-op run is saved -- the multiplayer menu shows Load and
// Abandon instead of Host then -- so the entry can never overwrite a co-op
// save the player has in progress.

using System;
using System.Reflection;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Platform;

namespace SpireMentor;

internal static class SimAiPartnerMenu
{
    private const string ButtonName = "SpireMentorAiPartnersButton";
    private const string Title = "With AI partners";
    private const string Description = "Co-op on this PC: an AI teammate plays a seat of its own. Pick your character and embark.";

    private static readonly FieldInfo? StackField =
        AccessTools.Field(typeof(NSubmenu), "_stack");

    public static void Install(Harmony patcher)
    {
        MethodInfo? ready = AccessTools.Method(typeof(NMultiplayerHostSubmenu), nameof(NMultiplayerHostSubmenu._Ready));
        if (ready == null)
        {
            Logger.Log("  patch skipped: SimAiPartnerMenu (NMultiplayerHostSubmenu._Ready not found)");
            return;
        }
        patcher.Patch(ready, postfix: new HarmonyMethod(typeof(SimAiPartnerMenu), nameof(ReadyPostfix)));
        Logger.Log("  patch applied: SimAiPartnerMenu (host menu entry)");
    }

    private static void ReadyPostfix(NMultiplayerHostSubmenu __instance)
    {
        try
        {
            AddButton(__instance);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartnerMenu] Could not add the button: {ex}");
        }
    }

    /// <summary>On unless <c>"ai_partners_menu": false</c> in
    /// SpireMentor.conf; the test lane's <c>--spire-mentor-ai-partners N</c>
    /// forces it on.</summary>
    internal static bool Enabled => SimConfig.Instance.AiPartnersMenu || SimMp.AiPartnersOverride > 0 || SimMp.AiPartnerJoinOverride > 0;

    private static void AddButton(NMultiplayerHostSubmenu submenu)
    {
        if (!Enabled || submenu.GetNodeOrNull(ButtonName) != null)
        {
            return;
        }
        NSubmenuButton? standard = submenu.GetNodeOrNull<NSubmenuButton>("StandardButton");
        NSubmenuButton? custom = submenu.GetNodeOrNull<NSubmenuButton>("CustomRunButton");
        if (standard == null || custom == null || standard.GetParent() is not Node parent)
        {
            Logger.Log("[SimAiPartnerMenu] The host menu has no Standard / Custom button; no entry.");
            return;
        }
        // Not the signals: the copy must not also start a Standard lobby.
        var button = (NSubmenuButton)standard.Duplicate((int)(Node.DuplicateFlags.Scripts | Node.DuplicateFlags.UseInstantiation));
        button.Name = ButtonName;
        // Its own hover shader: the material resource is shared otherwise,
        // and hovering one button would light up both.
        if (button.GetNodeOrNull<Control>("BgPanel") is { Material: { } material } bg)
        {
            bg.Material = (Material)material.Duplicate();
        }
        if (parent is Container)
        {
            parent.AddChild(button);
        }
        else
        {
            // The menu places its cards by hand: three 330 px columns, 400 px
            // apart, centred on 1920 (v0.111.0: Standard at 395, Custom at
            // 1195). A fourth column after Custom would leave the screen, so
            // all four are re-spaced, centred, in the same row.
            NSubmenuButton? daily = submenu.GetNodeOrNull<NSubmenuButton>("DailyButton");
            parent.AddChild(button);
            Control[] row = daily != null
                ? new Control[] { standard, daily, custom, button }
                : new Control[] { standard, custom, button };
            float width = standard.Size.X;
            float view = submenu.Size.X > 0 ? submenu.Size.X : submenu.GetViewportRect().Size.X;
            float gap = daily != null ? daily.Position.X - standard.Position.X - width : 70f;
            float step = width + gap;
            float start = (view - ((row.Length - 1) * step + width)) / 2f;
            if (start < 0)
            {
                // Narrower than the four fit at the game's gap: squeeze it.
                step = (view - width) / (row.Length - 1) * 0.98f;
                start = (view - ((row.Length - 1) * step + width)) / 2f;
            }
            for (int i = 0; i < row.Length; i++)
            {
                row[i].Position = new Vector2(start + i * step, standard.Position.Y);
            }
        }
        LogLayout(parent, standard, custom, button);
        button.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(_ => OnPressed(submenu)));
        Callable.From(() => SetTexts(button)).CallDeferred();
    }

    private static void SetTexts(NSubmenuButton button)
    {
        if (button.GetNodeOrNull<MegaLabel>("%Title") is { } title)
        {
            title.SetTextAutoSize(Title);
        }
        if (button.GetNodeOrNull<MegaRichTextLabel>("%Description") is { } description)
        {
            description.Text = Description;
        }
    }

    private static void LogLayout(Node parent, Control standard, Control custom, Control added)
    {
        Logger.Log($"[SimAiPartnerMenu] host menu: parent {parent.GetType().Name} '{parent.Name}'" +
                   (parent is Control pc ? $" size {pc.Size}" : "") +
                   $"; Standard at {standard.Position} size {standard.Size}; Custom at {custom.Position}; added at {added.Position}.");
    }

    private static void OnPressed(NMultiplayerHostSubmenu submenu)
    {
        try
        {
            NSubmenuStack? stack = StackField?.GetValue(submenu) as NSubmenuStack;
            if (stack == null)
            {
                Logger.Log("[SimAiPartnerMenu] No submenu stack; cannot host.");
                return;
            }
            SimAiPartnerSession.Start(stack, partnerCount: 1);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartnerMenu] Could not host with AI partners: {ex}");
        }
    }
}

/// <summary>
/// "AI partner N (Character)" for the partners, on any platform (Steam
/// knows no name for a partner's id). A session of its own runs on the null
/// platform, whose names are "Test Host" / "Test Client N": the host shows
/// the player's own Steam name there. Only the host's game runs this; a
/// human client without the mod names a partner however its platform does
/// (Steam: a blank or "[unknown]").
/// </summary>
internal static class SimAiPartnerNames
{
    public static void Install(Harmony patcher)
    {
        MethodInfo? raw = AccessTools.Method(typeof(PlatformUtil), nameof(PlatformUtil.GetPlayerNameRaw));
        if (raw == null)
        {
            Logger.Log("  patch skipped: SimAiPartnerNames (PlatformUtil.GetPlayerNameRaw not found)");
            return;
        }
        patcher.Patch(raw, postfix: new HarmonyMethod(typeof(SimAiPartnerNames), nameof(GetPlayerNameRawPostfix)));
        Logger.Log("  patch applied: SimAiPartnerNames");
    }

    private static void GetPlayerNameRawPostfix(PlatformType platformType, ulong playerId, ref string __result)
    {
        if (SimAiPartnerSession.Active is not { } session)
        {
            return;
        }
        try
        {
            foreach (SimAiPartner p in SimAiPartner.All)
            {
                if (p.NetId == playerId)
                {
                    __result = p.Character != null ? $"{p.Label} ({p.Character.Title.GetFormattedText()})" : p.Label;
                    return;
                }
            }
            if (platformType == PlatformType.None && !session.Mixed && playerId == session.HostNetId
                && LocalSteamName() is { Length: > 0 } name)
            {
                __result = name;
            }
        }
        catch (Exception)
        {
            // Keep the platform's name.
        }
    }

    private static string? LocalSteamName()
    {
        try
        {
            ulong me = PlatformUtil.GetLocalPlayerId(PlatformType.Steam);
            return me == 0 ? null : PlatformUtil.GetPlayerNameRaw(PlatformType.Steam, me);
        }
        catch (Exception)
        {
            return null;
        }
    }
}
