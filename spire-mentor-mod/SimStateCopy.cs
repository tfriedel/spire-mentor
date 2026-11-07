// SimStateCopy.cs -- put the current game state on the clipboard.
//
// The producer half of the workbench's live-state import: press the button,
// paste into the workbench, and get the same position back -- same seed, same
// deck, same map node, and (in a fight) the same hand, piles, powers and
// intents. From there it can be branched by hand or handed to SimLab.
//
// It emits exactly what the advisor already sends over the wire -- an
// `advice_request` built by the same builders (`SimCombatHandler.BuildCombatState`
// / `SimScreenAdvice.Describe`) -- rather than a bespoke "export" shape. One
// dialect means the client has one parser, and a state copied by hand is
// byte-for-byte a state the advisor could have sent.
//
// Every copy is ALSO written to `%LOCALAPPDATA%\SpireMentor\state-<ts>.json`.
// Clipboards are lossy (one slot, cleared by the next copy, and a big payload
// does not always survive), and a paper trail costs nothing.

using System;
using System.Collections.Generic;
using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

internal static class SimStateCopy
{
    /// <summary>Message type for a copy taken with no screen and no fight open.</summary>
    private const string RunSnapshotType = "run_snapshot";

    /// <summary>
    /// Copy the current state to the clipboard and to a timestamped file.
    /// Never throws: a failed copy notes why and leaves the clipboard alone.
    /// </summary>
    public static void Copy()
    {
        try
        {
            Dictionary<string, object>? state = BuildState();
            if (state == null)
            {
                SimActionLog.Note("Nothing to copy -- start a run first.");
                return;
            }
            string json = JsonSerializer.Serialize(state);
            DisplayServer.ClipboardSet(json);
            string? path = WriteSidecar(json);
            string where = path == null ? "" : $" (also saved to {path})";
            SimActionLog.Note(
                $"State copied to the clipboard{where}. Paste it into the workbench: " +
                "Import state.");
            Logger.Log($"[SimStateCopy] Copied {json.Length} chars, type={state.GetValueOrDefault("advice_for") ?? state.GetValueOrDefault("type")}");
        }
        catch (Exception ex)
        {
            SimActionLog.Note($"Could not copy the state: {ex.Message}");
            Logger.Log($"[SimStateCopy] Copy failed: {ex}");
        }
    }

    /// <summary>
    /// The state to copy, or null when there is no run.
    ///
    /// Preference order matches what the importer can do most with: the combat
    /// board (the only state that rebuilds a fight bit-exact), then the open
    /// screen, then a bare run snapshot. Every branch carries the same envelope
    /// (`SimStateExtractor.ApplyEnvelope`) -- seed, act, floor, run_state and
    /// the RNG cursors -- plus the map graph, which `WithMap` adds because the
    /// envelope leaves it off `combat_action` and `run_snapshot`. Without it the
    /// importer cannot place the run and restarts it at the act entrance.
    /// </summary>
    internal static Dictionary<string, object>? BuildState()
    {
        RunState? runState = RunManager.Instance?.DebugOnlyGetState();
        if (runState == null)
        {
            return null;
        }
        Player? player = LocalContext.GetMe(runState);

        if (player != null && (CombatManager.Instance?.IsInProgress ?? false))
        {
            Dictionary<string, object>? combat = SimCombatHandler.BuildCombatState(player);
            if (combat != null)
            {
                combat["type"] = "advice_request";
                combat["advice_for"] = "combat_action";
                // The brain the advisor lane asks with, so a pasted state
                // reproduces the request (PROTOCOL.md section 8).
                combat["brain"] = SimAdvisor.CurrentBrain;
                return WithMap(combat, runState);
            }
        }

        SimScreenDecision? screen = SimScreenAdvice.Describe();
        if (screen != null)
        {
            var msg = new Dictionary<string, object>(screen.State)
            {
                ["type"] = "advice_request",
                ["advice_for"] = screen.AdviceFor,
                ["brain"] = SimAdvisor.CurrentBrain,
            };
            return WithMap(msg, runState);
        }

        var snapshot = new Dictionary<string, object> { ["type"] = RunSnapshotType };
        SimStateExtractor.ApplyEnvelope(snapshot, runState, player);
        return WithMap(snapshot, runState);
    }

    /// <summary>
    /// Add the act map graph (and with it the current node) when the builder
    /// left it off. `ApplyEnvelope` omits it from `combat_action` on purpose --
    /// that message goes out ~4x/s and the graph walk is not free -- but a copy
    /// is one-shot, and the map is what places the imported run where it was.
    /// </summary>
    private static Dictionary<string, object> WithMap(Dictionary<string, object> msg, RunState runState)
    {
        if (!msg.ContainsKey("map"))
        {
            try { msg["map"] = SimStateExtractor.SerializeMapGraph(runState); }
            catch (Exception ex) { Logger.Log($"[SimStateCopy] Map not copied: {ex.Message}"); }
        }
        return msg;
    }

    /// <summary>
    /// Write the payload beside the advisor's own logs; returns the path, or
    /// null if it could not be written. Best-effort by design -- the clipboard
    /// is the deliverable, this is the backup.
    /// </summary>
    private static string? WriteSidecar(string json)
    {
        try
        {
            // Fully qualified: `Godot` is in scope and ships its own
            // `Environment` (the 3D world one), so a bare `Environment` here
            // is ambiguous — the same reason SimWorkbenchLink qualifies it.
            string dir = System.IO.Path.Combine(
                System.Environment.GetFolderPath(System.Environment.SpecialFolder.LocalApplicationData),
                "SpireMentor");
            System.IO.Directory.CreateDirectory(dir);
            string path = System.IO.Path.Combine(
                dir, $"state-{DateTime.Now:yyyyMMdd-HHmmss}.json");
            System.IO.File.WriteAllText(path, json);
            return path;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimStateCopy] Sidecar write failed: {ex.Message}");
            return null;
        }
    }
}
