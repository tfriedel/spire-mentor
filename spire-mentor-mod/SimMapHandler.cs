// SimMapHandler.cs -- agent-driven map navigation (map_select).
// Originally ported from zhiyue/sts2-rl-agent (STS2BridgeMod); rewritten clean-room.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

public class SimMapHandler : IScreenHandler, IHandler
{
    private const string RunNodePath = "/root/Game/RootSceneContainer/Run";
    private const int NodeAppearWaitMs = 3000;

    // The game's own MapScreenHandler bounds the point-enable and room-enter
    // waits at 30 s (AutoSlayConfig.mapPointEnabledTimeout / roomEnterTimeout).
    // Hard-coded here rather than read off AutoSlayConfig: roomEnterTimeout
    // only exists from v0.113.0 and mapPointEnabledTimeout from v0.111.0, and
    // the one shipped DLL must bind against v0.107.1 too.
    private static readonly TimeSpan PointEnableWait = TimeSpan.FromSeconds(30);
    private static readonly TimeSpan RoomEnterWait = TimeSpan.FromSeconds(30);

    /// <summary>Completed by <see cref="OnRoomEntered"/> while a map click is in flight.</summary>
    private TaskCompletionSource<bool>? _roomEntered;

    public Type ScreenType => typeof(NMapScreen);

    public TimeSpan Timeout => TimeSpan.FromHours(24);

    /// <summary>
    /// Asks the agent which reachable node to travel to, clicks it, and
    /// returns once the next room has been entered.
    /// </summary>
    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        Logger.Log("[SimMap] Waiting for the map screen");
        NRun runNode = SimRoomNodes.SceneRoot.GetNode<NRun>(RunNodePath);
        await WaitHelper.Until(() => runNode.GlobalUi.MapScreen.IsVisibleInTree(), ct,
            AutoSlayConfig.mapScreenTimeout, "Map screen never became visible");
        NMapScreen map = runNode.GlobalUi.MapScreen;

        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        // The new act's nodes can lag the screen by a few frames.
        List<NMapPoint> available = new List<NMapPoint>();
        for (int waitedMs = 0; runState != null; waitedMs += 16)
        {
            available = AvailableNodes(map, runState);
            if (available.Count > 0 || waitedMs >= NodeAppearWaitMs)
                break;
            await Task.Delay(16, ct);
        }
        if (available.Count == 0)
        {
            Logger.Log("[SimMap] No reachable map nodes; nothing to choose");
            return;
        }

        Player? player = LocalContext.GetMe(runState);
        List<Dictionary<string, object>> nodes = available.Select((p, i) => new Dictionary<string, object>
        {
            ["index"] = i,
            ["type"] = p.Point.PointType.ToString(),
            ["row"] = p.Point.coord.row,
            ["col"] = p.Point.coord.col,
        }).ToList();
        var message = new Dictionary<string, object>
        {
            [NonCombatBridgeProtocol.TypeField] = "map_select",
            ["nodes"] = nodes,
            ["map"] = SimStateExtractor.SerializeMapGraph(runState!),
        };

        NMapPoint point = await ChooseNodeAsync(available, message, runState, player, ct);

        // Points stay disabled while the map animates in; a post-boss act
        // transition can enable travel several seconds late (AutoSlayConfig
        // comment on mapPointEnabledTimeout), so this uses the game's own 30 s
        // bound rather than a tighter one that would abort a slow transition.
        await WaitHelper.Until(() => point.IsEnabled, ct, PointEnableWait,
            "Chosen map point never became enabled");

        // Default options on purpose: the continuation runs inline inside the
        // RoomEntered callback, the timing proven against the real game.
        RunManager.Instance.RoomEntered += OnRoomEntered;
        try
        {
            await ClickAndAwaitRoomAsync(map, point, ct);
        }
        finally
        {
            RunManager.Instance.RoomEntered -= OnRoomEntered;
            _roomEntered = null;
        }
        Logger.Log("[SimMap] Entered the next room");
    }

    /// <summary>
    /// Clicks the chosen node and waits for the room to be entered, bounded at
    /// <see cref="RoomEnterWait"/> (30 s, matching the game's own
    /// roomEnterTimeout) -- not the 10 s map-screen bound, which abandoned a
    /// travel a slow host (or an attached run's extra overlay frames) finished
    /// in >10 s but <30 s, three times too early (GH #55).
    ///
    /// Recovery for one specific stall: an attach can land with the map open
    /// but <c>IsTravelEnabled</c> never set -- the node renders Travelable and
    /// its control is enabled, so it is clicked, but the game refuses travel
    /// (<c>NMapPoint.IsTravelable</c> reads the screen flag live) and no room
    /// is entered. When the 30 s wait times out AND travel is off, re-arm it
    /// and click once more. This fires only after a genuine stall that is
    /// already abandoning the run, so it never races the game's own travel
    /// gating during normal play (an act transition, LordsParasol). A plain
    /// <c>SetTravelEnabled(true)</c>, not <c>RunManager.Proceed()</c>: the
    /// latter advances a Blitz node and re-opens the map as side effects.
    /// </summary>
    private async Task ClickAndAwaitRoomAsync(NMapScreen map, NMapPoint point, CancellationToken ct)
    {
        _roomEntered = new TaskCompletionSource<bool>();
        await UiHelper.Click(point);
        try
        {
            await WaitHelper.ForTask(_roomEntered.Task, ct, RoomEnterWait,
                "Next room was not entered after clicking the map point");
            return;
        }
        // System.TimeoutException, not the framework's own timeout type: that
        // was AutoSlayTimeoutException up to v0.111.0 and DriverTimeoutException
        // from v0.113.0 on, and both derive from TimeoutException, so catching
        // the base binds the one shipped DLL against every supported build.
        catch (TimeoutException)
        {
            Logger.Log($"[SimMap] Room not entered after click: node=({point.Point.coord.row},{point.Point.coord.col}) " +
                       $"state={point.State} screenTravelEnabled={map.IsTravelEnabled}");
            if (map.IsTravelEnabled)
                throw; // travel was on; the click itself did not land -- not ours to re-arm here
        }

        Logger.Log("[SimMap] Travel was disabled under an open map; re-enabling and retrying the click once");
        map.SetTravelEnabled(true);
        _roomEntered = new TaskCompletionSource<bool>();
        await UiHelper.Click(point);
        await WaitHelper.ForTask(_roomEntered.Task, ct, RoomEnterWait,
            "Next room was not entered after re-enabling travel and re-clicking the map point");
    }

    /// <summary>
    /// The map points the player can travel to next: row 0 before the first
    /// move, otherwise the points reachable from the last visited one. Shared with
    /// advisor mode so both lanes offer the same set.
    ///
    /// Only nodes of the act being played: right after an act change the
    /// previous act's start/boss nodes are still in the tree
    /// (NMapScreen.RemoveAllMapPointsAndPaths frees them deferred), so its
    /// Ancient showed up next to the new one. Picking the stale one waited
    /// for a point that never enables and ended the run (GH #1287, Silent
    /// seed 1000 act 2; Ironclad act 3 was offered (3,0) twice).
    /// </summary>
    internal static List<NMapPoint> AvailableNodes(Node mapScreen, RunState runState)
    {
        List<NMapPoint> points = UiHelper.FindAll<NMapPoint>(mapScreen)
            .Where(p => GodotObject.IsInstanceValid(p) && !p.IsQueuedForDeletion()
                && p.Point != null && ReferenceEquals(runState.Map.GetPoint(p.Point.coord), p.Point))
            .ToList();
        if (runState.VisitedMapCoords.Count == 0)
        {
            return points.Where(p => p.Point.coord.row == 0).ToList();
        }

        MapCoord current = runState.VisitedMapCoords[runState.VisitedMapCoords.Count - 1];
        NMapPoint? here = points.FirstOrDefault(p => p.Point.coord.Equals(current));
        if (here == null)
        {
            return new List<NMapPoint>();
        }

        // The game's rule (NMapScreen.RecalculateTravelability): the boss
        // follows the last row through Children; any other point goes through
        // MapTravel, which opens the whole next row while Winged Boots has
        // charges (GH #1287, Ironclad seed 1005 was offered only the children).
        IEnumerable<MapPoint> reachable = current.row >= runState.Map.GetRowCount() - 1
            ? here.Point.Children
            : MapTravel.GetTravelablePointsFrom(runState, here.Point);
        var next = new HashSet<MapCoord>(reachable.Select(point => point.coord));
        return points.Where(p => next.Contains(p.Point.coord)).ToList();
    }

    /// <summary>
    /// Asks until the reply names a valid node. Every ask re-envelopes the
    /// message (fresh state_id, PROTOCOL.md section 3a); the reply's
    /// <c>index</c> is read without an <c>action</c> check.
    /// </summary>
    private static async Task<NMapPoint> ChooseNodeAsync(
        List<NMapPoint> available, Dictionary<string, object> message, RunState? runState, Player? player,
        CancellationToken ct)
    {
        while (true)
        {
            SimStateExtractor.ApplyEnvelope(message, runState, player);
            long stateId = SimAgentChoice.ReadStateId(message);
            string json = JsonSerializer.Serialize(message);
            string reply = await SimAgentGate.RequestActionAsync(json, ct, "map_select");
            try
            {
                if (!SimProtocolHelpers.TryParse(reply, runState, player, out JsonElement root) ||
                    !SimProtocolHelpers.CheckStateId(root, stateId, runState, player))
                {
                    continue;
                }

                if (!root.TryGetProperty("index", out JsonElement indexEl) ||
                    indexEl.ValueKind != JsonValueKind.Number ||
                    !indexEl.TryGetInt32(out int index))
                {
                    SimProtocolHelpers.SendError("malformed_action",
                        "map_select reply requires an integer \"index\"", runState, player);
                    continue;
                }

                if (index >= 0 && index < available.Count)
                {
                    MapPoint chosen = available[index].Point;
                    Logger.Log($"[SimMap] Agent chose node {index}: {chosen.PointType} " +
                               $"(row {chosen.coord.row}, col {chosen.coord.col})");
                    return available[index];
                }

                SimProtocolHelpers.SendError("invalid_index",
                    $"index {index} out of range ({available.Count} nodes)", runState, player);
                Logger.Log($"[SimMap] Invalid node index {index} for {available.Count} nodes; asking again");
            }
            catch (Exception ex)
            {
                Logger.Log($"[SimMap] Error handling map_select reply: {ex.Message}");
                SimProtocolHelpers.SendError("internal_error", ex.Message, runState, player);
            }
        }
    }

    private void OnRoomEntered()
    {
        _roomEntered?.TrySetResult(true);
    }
}
