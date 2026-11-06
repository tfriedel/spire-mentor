// SimAiPartnerLobby.cs -- "Add AI partner" / "Remove AI partner" in the host's
// co-op lobby (docs/plans/2026-11-05-ai-coop-partners.md, phase 4).
//
// Two tiles in the lobby's player list, next to the game's own Invite tile
// (NRemoteLobbyPlayerContainer), on the host only. In a lobby the player
// hosts the game's way (Steam; fastmp's ENet) the first press puts the
// partners next to the game's host (SimAiPartnerSession.ForLobby,
// SimMixedHost), so friends can still join over Steam: humans and AI
// partners in one run. In a session of its own ("With AI partners") the
// tiles add and drop partners there.
//
// The tiles are copies of the Invite tile, so they look like the game's
// own. The copy keeps the Invite button's script (its hover shader, its
// click handling); three prefixes turn its invite behaviour off for the
// copies only: the click (OnRelease), the map hotkey it answers to
// (Hotkeys), and the visibility it would set from the platform
// (UpdateVisibility).

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Game.Lobby;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

internal static class SimAiPartnerLobby
{
    private const string AddName = "SpireMentorAddAiPartner";
    private const string RemoveName = "SpireMentorRemoveAiPartner";
    private const string AddText = "Add AI partner";
    private const string RemoveText = "Remove AI partner";
    private const string PickerName = "SpireMentorAiPartnerPicker";

    private static readonly FieldInfo? LobbyField = AccessTools.Field(typeof(NRemoteLobbyPlayerContainer), "_lobby");
    private static readonly FieldInfo? NodesField = AccessTools.Field(typeof(NRemoteLobbyPlayerContainer), "_nodes");
    private static readonly FieldInfo? ListField = AccessTools.Field(typeof(NRemoteLobbyPlayerContainer), "_container");
    private static readonly FieldInfo? InviteField = AccessTools.Field(typeof(NRemoteLobbyPlayerContainer), "_inviteButton");

    /// <summary>Our copies of the Invite button (by instance id), and what a
    /// press does.</summary>
    private static readonly Dictionary<ulong, Action> Tiles = new();

    /// <summary>The lobbies (host services) the lane's
    /// <c>--spire-mentor-ai-partner-join N</c> already filled.</summary>
    private static readonly HashSet<NetHostGameService> AutoJoined = new(ReferenceEqualityComparer.Instance);

    public static void Install(Harmony patcher)
    {
        Type container = typeof(NRemoteLobbyPlayerContainer);
        Type invite = typeof(NInvitePlayersButton);
        Patch(patcher, AccessTools.Method(container, nameof(NRemoteLobbyPlayerContainer.Initialize)), postfix: nameof(ContainerChangedPostfix));
        Patch(patcher, AccessTools.Method(container, nameof(NRemoteLobbyPlayerContainer.OnPlayerConnected)), postfix: nameof(ContainerChangedPostfix));
        Patch(patcher, AccessTools.Method(container, nameof(NRemoteLobbyPlayerContainer.OnPlayerDisconnected)), postfix: nameof(ContainerChangedPostfix));
        Patch(patcher, AccessTools.Method(container, nameof(NRemoteLobbyPlayerContainer.Cleanup)), postfix: nameof(ContainerChangedPostfix));
        Patch(patcher, AccessTools.Method(invite, "OnRelease"), prefix: nameof(OnReleasePrefix));
        Patch(patcher, AccessTools.PropertyGetter(invite, "Hotkeys"), prefix: nameof(HotkeysPrefix));
        Patch(patcher, AccessTools.Method(invite, "UpdateVisibility"), prefix: nameof(UpdateVisibilityPrefix));
    }

    private static void Patch(Harmony patcher, MethodBase? target, string? prefix = null, string? postfix = null)
    {
        if (target == null)
        {
            Logger.Log($"  patch skipped: SimAiPartnerLobby {prefix ?? postfix} (target not found)");
            return;
        }
        try
        {
            patcher.Patch(target,
                prefix: prefix != null ? new HarmonyMethod(typeof(SimAiPartnerLobby), prefix) : null,
                postfix: postfix != null ? new HarmonyMethod(typeof(SimAiPartnerLobby), postfix) : null);
            Logger.Log($"  patch applied: SimAiPartnerLobby {target.DeclaringType?.Name}.{target.Name}");
        }
        catch (Exception ex)
        {
            Logger.Log($"  patch skipped: SimAiPartnerLobby {target.DeclaringType?.Name}.{target.Name} ({ex.Message})");
        }
    }

    private static bool IsTile(NInvitePlayersButton button) =>
        GodotObject.IsInstanceValid(button) && Tiles.ContainsKey(button.GetInstanceId());

    private static bool OnReleasePrefix(NInvitePlayersButton __instance)
    {
        if (!IsTile(__instance))
        {
            return true;
        }
        try
        {
            Tiles[__instance.GetInstanceId()]();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartnerLobby] {ex}");
        }
        return false;
    }

    private static bool HotkeysPrefix(NInvitePlayersButton __instance, ref string[] __result)
    {
        if (!IsTile(__instance))
        {
            return true;
        }
        __result = Array.Empty<string>();
        return false;
    }

    private static bool UpdateVisibilityPrefix(NInvitePlayersButton __instance) => !IsTile(__instance);

    private static void ContainerChangedPostfix(NRemoteLobbyPlayerContainer __instance)
    {
        try
        {
            Refresh(__instance);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartnerLobby] {ex}");
        }
    }

    /// <summary>The host's service of the container's lobby, when the tiles
    /// belong there: a Standard lobby this game hosts.</summary>
    private static NetHostGameService? HostedLobby(NRemoteLobbyPlayerContainer container, out StartRunLobby? lobby)
    {
        lobby = LobbyField?.GetValue(container) as StartRunLobby;
        if (!SimAiPartnerMenu.Enabled || lobby == null || lobby.GameMode != GameMode.Standard
            || lobby.NetService is not NetHostGameService host || host.Type != NetGameType.Host || !host.IsConnected)
        {
            return null;
        }
        return host;
    }

    private static void Refresh(NRemoteLobbyPlayerContainer container)
    {
        NetHostGameService? host = HostedLobby(container, out StartRunLobby? lobby);
        if (host == null && !HasTiles(container))
        {
            return;
        }
        (Control add, Control remove)? tiles = host != null ? EnsureTiles(container, host) : FindTiles(container);
        if (tiles is not var (addTile, removeTile))
        {
            return;
        }
        SimAiPartnerSession? session = SimAiPartnerSession.Active is { } s && host != null && ReferenceEquals(s.Service, host) ? s : null;
        int partners = session?.Partners.Count ?? 0;
        int players = lobby != null ? SimCoopDummy.LobbyPlayerCount(lobby) : 0;
        addTile.Visible = host != null && players < SimAiPartnerSession.MaxPlayers && partners < SimAiPartner.NetIds.Length
                          && !lobby!.IsAboutToBeginGame();
        removeTile.Visible = host != null && partners > 0 && !lobby!.IsAboutToBeginGame();
        Reorder(container, addTile, removeTile);
        if (host != null && session != null)
        {
            WirePartnerSlots(container, session);
        }

        if (host != null && SimMp.AiPartnerJoinOverride > 0 && AutoJoined.Add(host))
        {
            AutoJoin(container, host);
        }
    }

    private static bool HasTiles(NRemoteLobbyPlayerContainer container) => FindTiles(container) != null;

    private static (Control, Control)? FindTiles(NRemoteLobbyPlayerContainer container)
    {
        if (ListField?.GetValue(container) is not Container list)
        {
            return null;
        }
        return list.GetNodeOrNull<Control>(AddName) is { } add && list.GetNodeOrNull<Control>(RemoveName) is { } remove
            ? (add, remove)
            : null;
    }

    private static (Control, Control)? EnsureTiles(NRemoteLobbyPlayerContainer container, NetHostGameService host)
    {
        if (FindTiles(container) is { } found)
        {
            return found;
        }
        if (ListField?.GetValue(container) is not Container list
            || InviteField?.GetValue(container) is not NInvitePlayersButton invite
            || invite.GetParent() is not Control inviteTile)
        {
            Logger.Log("[SimAiPartnerLobby] The lobby has no Invite tile to copy; no AI-partner tiles.");
            return null;
        }
        Control add = CopyTile(list, inviteTile, AddName, AddText, () => OnAdd(container));
        Control remove = CopyTile(list, inviteTile, RemoveName, RemoveText, () => OnRemove(container));
        Logger.Log("[SimAiPartnerLobby] Added the AI-partner tiles to the lobby.");
        return (add, remove);
    }

    private static Control CopyTile(Container list, Control inviteTile, string name, string text, Action press)
    {
        // The script comes along (hover, click); the signals do not, so the
        // copy cannot open the invite dialog through a connection.
        var tile = (Control)inviteTile.Duplicate((int)(Node.DuplicateFlags.Scripts | Node.DuplicateFlags.UseInstantiation));
        tile.Name = name;
        NInvitePlayersButton? button = tile as NInvitePlayersButton
                                        ?? tile.FindChildren("*", "", recursive: true, owned: false).OfType<NInvitePlayersButton>().FirstOrDefault();
        if (button == null)
        {
            throw new InvalidOperationException("the Invite tile has no NInvitePlayersButton");
        }
        // Its own hover shader: the material is shared otherwise.
        if (button.GetNodeOrNull<Control>("Background") is { Material: { } material } bg)
        {
            bg.Material = (Material)material.Duplicate();
        }
        Tiles[button.GetInstanceId()] = press;
        button.TreeExited += () => Tiles.Remove(button.GetInstanceId());
        list.AddChild(tile);
        // The button's _Ready names it "Invite"; rename after it.
        Callable.From(() =>
        {
            if (GodotObject.IsInstanceValid(button) && button.GetNodeOrNull<MegaRichTextLabel>("Label") is { } label)
            {
                label.SetTextAutoSize(text);
            }
        }).CallDeferred();
        return tile;
    }

    /// <summary>
    /// The character picker: an AI partner's own slot in the player list
    /// (the game's NRemoteLobbyPlayer) gets a transparent click target on
    /// top. Left-click: its next character, right-click: the previous one
    /// (SimAiPartner.CycleCharacter); the slot's character label and icon
    /// follow through the game's own OnPlayerChanged. The nameplate says so,
    /// since the game has no tooltips to put it in. A load lobby's slots are
    /// a different container (NRemoteLoadLobbyPlayerContainer): a saved
    /// run's characters are fixed.
    /// </summary>
    private static void WirePartnerSlots(NRemoteLobbyPlayerContainer container, SimAiPartnerSession session)
    {
        if (NodesField?.GetValue(container) is not List<NRemoteLobbyPlayer> nodes)
        {
            return;
        }
        foreach (NRemoteLobbyPlayer slot in nodes)
        {
            SimAiPartner? partner = session.Partners.FirstOrDefault(p => p.NetId == slot.PlayerId);
            if (partner == null || !GodotObject.IsInstanceValid(slot) || slot.GetNodeOrNull(PickerName) != null)
            {
                continue;
            }
            var picker = new Control
            {
                Name = PickerName,
                MouseFilter = Control.MouseFilterEnum.Stop,
                MouseDefaultCursorShape = Control.CursorShape.PointingHand,
            };
            picker.SetAnchorsPreset(Control.LayoutPreset.FullRect);
            ulong netId = partner.NetId;
            picker.Connect(Control.SignalName.GuiInput, Callable.From<InputEvent>(ev => OnPickerInput(container, picker, netId, ev)));
            slot.AddChild(picker);
            string label = partner.Label;
            Callable.From(() =>
            {
                // After the slot's _Ready, which names it from the platform.
                if (GodotObject.IsInstanceValid(slot) && slot.GetNodeOrNull<MegaLabel>("%NameplateLabel") is { } name)
                {
                    name.SetTextAutoSize($"{label} (click: character)");
                }
            }).CallDeferred();
            Logger.Log($"[SimAiPartnerLobby] {label}'s slot picks its character on click.");
        }
    }

    private static void OnPickerInput(NRemoteLobbyPlayerContainer container, Control picker, ulong netId, InputEvent ev)
    {
        if (ev is not InputEventMouseButton { Pressed: true } mb
            || (mb.ButtonIndex != MouseButton.Left && mb.ButtonIndex != MouseButton.Right))
        {
            return;
        }
        picker.AcceptEvent();
        try
        {
            if (HostedLobby(container, out StartRunLobby? lobby) == null || lobby == null || lobby.IsAboutToBeginGame())
            {
                return;
            }
            SimAiPartner? partner = SimAiPartner.All.FirstOrDefault(p => p.NetId == netId);
            partner?.CycleCharacter(mb.ButtonIndex == MouseButton.Left ? 1 : -1);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartnerLobby] character pick: {ex}");
        }
    }

    /// <summary>The lane's picker check (<c>--spire-mentor-ai-partner-pick
    /// N</c>): N left clicks at the centre of the partner's slot, pushed into
    /// the viewport as a player's mouse would be, so the GUI's own routing
    /// decides who gets them.</summary>
    internal static async System.Threading.Tasks.Task ClickSlotAsync(SceneTree tree, ulong netId, int clicks)
    {
        Control? picker = null;
        for (int i = 0; i < 120 && picker == null; i++)
        {
            picker = tree.Root.FindChildren(PickerName, "", recursive: true, owned: false)
                .OfType<Control>()
                .FirstOrDefault(c => c.GetParent() is NRemoteLobbyPlayer slot && slot.PlayerId == netId && c.IsVisibleInTree());
            if (picker == null)
            {
                await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
            }
        }
        if (picker == null)
        {
            Logger.Log($"[SimAiPartnerLobby] pick lane: no picker on {netId}'s slot.");
            return;
        }
        Vector2 at = picker.GetGlobalRect().GetCenter();
        Logger.Log($"[SimAiPartnerLobby] pick lane: {clicks} click(s) at {at} on {netId}'s slot ({picker.GetGlobalRect()}).");
        for (int i = 0; i < clicks; i++)
        {
            foreach (bool pressed in new[] { true, false })
            {
                var ev = new InputEventMouseButton
                {
                    ButtonIndex = MouseButton.Left,
                    Pressed = pressed,
                    Position = at,
                    GlobalPosition = at,
                    ButtonMask = pressed ? MouseButtonMask.Left : 0,
                };
                tree.Root.PushInput(ev, inLocalCoords: true);
                await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
            }
            await System.Threading.Tasks.Task.Delay(300);
        }
    }

    /// <summary>The game keeps its Invite tile last; ours go just before it.</summary>
    private static void Reorder(NRemoteLobbyPlayerContainer container, Control add, Control remove)
    {
        if (ListField?.GetValue(container) is not Container list)
        {
            return;
        }
        list.MoveChild(add, -1);
        list.MoveChild(remove, -1);
        if (InviteField?.GetValue(container) is NInvitePlayersButton { } invite && invite.GetParent() is Control inviteTile
            && inviteTile.GetParent() == list)
        {
            list.MoveChild(inviteTile, -1);
        }
    }

    private static void OnAdd(NRemoteLobbyPlayerContainer container)
    {
        if (HostedLobby(container, out _) is not { } host)
        {
            return;
        }
        SimAiPartnerSession? session = SimAiPartnerSession.ForLobby(host);
        if (session == null)
        {
            Logger.Log("[SimAiPartnerLobby] This lobby cannot take AI partners (no game host to join).");
            return;
        }
        SimAiPartner? partner = session.Add(null, container);
        Logger.Log(partner != null ? $"[SimAiPartnerLobby] {partner.Label} is joining." : "[SimAiPartnerLobby] No free AI-partner seat.");
        Refresh(container);
    }

    private static void OnRemove(NRemoteLobbyPlayerContainer container)
    {
        if (HostedLobby(container, out _) is { } host && SimAiPartnerSession.Active is { } s && ReferenceEquals(s.Service, host))
        {
            s.RemoveLast();
        }
        Refresh(container);
    }

    /// <summary>The mixed lane: <c>--spire-mentor-ai-partner-join N</c>
    /// presses "Add AI partner" N times once the lobby is up.</summary>
    private static void AutoJoin(NRemoteLobbyPlayerContainer container, NetHostGameService host)
    {
        SimAiPartnerSession? session = SimAiPartnerSession.ForLobby(host);
        if (session == null)
        {
            return;
        }
        string[] chars = (SimMp.AiPartnerCharacters ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        int n = SimMp.AiPartnerJoinOverride;
        Logger.Log($"[SimAiPartnerLobby] --spire-mentor-ai-partner-join {n}: adding to the game's own lobby.");
        for (int i = session.Partners.Count; i < n; i++)
        {
            session.Add(i < chars.Length ? chars[i] : null, container);
        }
        Refresh(container);
    }
}
