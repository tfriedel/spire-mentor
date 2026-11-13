// SimCoopDummy.cs -- readies a hands-free instance in a co-op lobby.
//
// The co-op test lane (tools/coop_test.ps1) runs two game instances on one
// machine with the game's own `--fastmp` dev launch: it opens the lobby and,
// for a client, joins 127.0.0.1. What it does not do is pick a character and
// press Embark, which is the only step between the lobby and a run. This does
// it for an instance launched with `--spire-mentor-autoplay`; once the run is
// live, SimAdvisorAutoplay hands it to the advisor lane, the same as F8 + F7.
//
// A host only embarks once the whole party is in the lobby
// (`--spire-mentor-coop-players N`, default 2): with fewer, the run would
// start without the players still joining.
//
// The AI-partner lane (tools/ai_partner_test.ps1) uses it too: one instance
// hosts with `--spire-mentor-ai-partners N` (SimAiPartnerSession) instead of
// `--fastmp`, and this embarks once the partners are in.

using System;
using System.Collections;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Game.Lobby;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace SpireMentor;

internal static class SimCoopDummy
{
    private const double PollSeconds = 1.0;
    private const string ConfirmButtonPath = "ConfirmButton";
    private const string CharButtonContainerPath = "CharSelectButtons/ButtonContainer";

    private static bool _pressing;

    public static void InstallIfRequested()
    {
        if (!(SimMp.FastMpLaunch || SimMp.AiPartnersOverride > 0) || !SimMp.AutoplayOverride)
        {
            return;
        }
        try
        {
            if (Engine.GetMainLoop() is not SceneTree tree)
            {
                return;
            }
            var timer = new Godot.Timer
            {
                Name = "SpireMentorCoopDummy",
                WaitTime = PollSeconds,
                Autostart = true,
                OneShot = false,
            };
            timer.Connect(Godot.Timer.SignalName.Timeout, Callable.From(() => Poll(tree)));
            tree.Root.AddChildSafely(timer);
            Logger.Log("[SimCoopDummy] Will ready up in the co-op lobby.");
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCoopDummy] Could not install: {ex.Message}");
        }
    }

    private static void Poll(SceneTree tree)
    {
        if (_pressing || RunManager.Instance.IsInProgress)
        {
            return;
        }
        try
        {
            Control? screen = LobbyScreen(tree.Root);
            if (screen == null)
            {
                return;
            }
            StartRunLobby? lobby = screen.GetType().GetProperty("Lobby")?.GetValue(screen) as StartRunLobby;
            NetGameType? type = lobby?.NetService?.Type;
            if (type != NetGameType.Client && !(type == NetGameType.Host && LobbyPlayerCount(lobby!) >= SimMp.CoopPlayers))
            {
                return;
            }
            NClickableControl? confirm = screen.GetNodeOrNull<NClickableControl>(ConfirmButtonPath);
            if (confirm == null || !confirm.IsEnabled || !confirm.IsVisibleInTree())
            {
                return;
            }
            _pressing = true;
            TaskHelper.RunSafely(EmbarkAsync(screen, confirm));
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCoopDummy] Poll: {ex.Message}");
        }
    }

    /// <summary>The lobby screens a co-op run starts from: character select
    /// (Standard), and since v0.113.0 NBlitzScreen / NDreadScreen, which are
    /// not character selects but have the same Lobby, ConfirmButton and
    /// CharSelectButtons nodes. Those two are reached by name: the mod
    /// compiles against builds that do not have them. They sit next to
    /// character select (Screens.Blitz / Screens.Dread), so their names are
    /// built from its namespace.</summary>
    private static readonly Type[] LobbyScreenTypes = new[]
    {
        typeof(NCharacterSelectScreen),
        SiblingScreenType("Blitz", "NBlitzScreen"),
        SiblingScreenType("Dread", "NDreadScreen"),
    }.OfType<Type>().ToArray();

    private static Type? SiblingScreenType(string folder, string name)
    {
        string ns = typeof(NCharacterSelectScreen).Namespace ?? "";
        int dot = ns.LastIndexOf('.');
        return dot < 0 ? null : AccessTools.TypeByName($"{ns[..dot]}.{folder}.{name}");
    }

    /// <summary>The visible lobby screen, if one is open.</summary>
    private static Control? LobbyScreen(Node node)
    {
        if (node is Control control && LobbyScreenTypes.Any(t => t.IsInstanceOfType(node)))
        {
            return control.IsVisibleInTree() ? control : null;
        }
        foreach (Node child in node.GetChildren())
        {
            if (LobbyScreen(child) is { } found)
            {
                return found;
            }
        }
        return null;
    }

    private static async Task EmbarkAsync(Control screen, NClickableControl confirm)
    {
        try
        {
            SelectCharacter(screen);
            // A fresh profile (`--force-steam off` may not share the Steam
            // one) would stop at the tutorials prompt before readying.
            SaveManager.Instance.MarkFtueAsComplete("accept_tutorials_ftue");
            SaveManager.Instance.SetFtuesEnabled(enabled: false);
            await Task.Delay(500);
            Logger.Log("[SimCoopDummy] Embarking (ready in the co-op lobby).");
            await UiHelper.Click(confirm);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCoopDummy] Embark failed: {ex.Message}");
        }
        finally
        {
            _pressing = false;
        }
    }

    /// <summary><c>--spire-mentor-character NAME</c>, else the screen's own
    /// preselection.</summary>
    private static void SelectCharacter(Control screen)
    {
        string? wanted = SimMp.CharacterOverride;
        if (string.IsNullOrEmpty(wanted))
        {
            return;
        }
        Control? container = screen.GetNodeOrNull<Control>(CharButtonContainerPath);
        if (container == null)
        {
            return;
        }
        foreach (NCharacterSelectButton button in UiHelper.FindAll<NCharacterSelectButton>(container))
        {
            if (!button.IsLocked && button.Character.Id.Entry.Contains(wanted, StringComparison.OrdinalIgnoreCase))
            {
                Logger.Log($"[SimCoopDummy] Selecting {button.Character.Id.Entry}.");
                button.Select();
                return;
            }
        }
        Logger.Log($"[SimCoopDummy] No unlocked character matches \"{wanted}\"; keeping the preselection.");
    }

    /// <summary>The lobby's player count. Read by reflection: the element
    /// type is LobbyPlayer in v0.107.1 and StartRunLobbyPlayer in v0.111.0,
    /// so the property's signature differs between the builds.</summary>
    internal static int LobbyPlayerCount(StartRunLobby lobby) =>
        lobby.GetType().GetProperty("Players")?.GetValue(lobby) is ICollection players ? players.Count : 0;
}
