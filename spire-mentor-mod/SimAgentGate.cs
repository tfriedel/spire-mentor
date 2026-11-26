// SimAgentGate.cs -- blocking "ask the agent, never guess" helper.
//
// New file, not present in the donor (zhiyue/sts2-rl-agent bridge_mod). The
// donor fell back to *random* play after a 30s agent timeout or when no
// client was connected -- this consumes game RNG and silently perturbs
// determinism/parity. Required deviation #3 in the bridge plan explicitly
// forbids that for SpireMentor: every handler that used to do
// `random.NextItem(options)` on a timeout/disconnect now instead calls
// through this gate, which blocks (waiting for a client to connect, then for
// a correlated response) until the agent actually answers. It never invents
// a choice on the mod's own initiative.
//
// agent_timeout_seconds (SimConfig) is *not* a give-up timeout: it only
// controls how often we re-poll/re-log while waiting (0 = one indefinite
// wait). On expiry we log and loop again rather than falling back.

using System;
using System.Threading;
using System.Threading.Tasks;

namespace SpireMentor;

internal static class SimAgentGate
{
    private static TimeSpan PerAttemptTimeout =>
        SimConfig.Instance.AgentTimeoutSeconds > 0
            ? TimeSpan.FromSeconds(SimConfig.Instance.AgentTimeoutSeconds)
            : TimeSpan.FromHours(24);

    /// <summary>
    /// Serializes concurrent requesters. BridgeServer holds exactly ONE
    /// pending (request_id, TaskCompletionSource) slot and each new
    /// SendStateAndWaitForActionAsync CANCELS the previous requester's wait
    /// -- so two concurrent callers (e.g. the instant-mode combat handler
    /// snapshotting the next turn while the game is still blocked inside
    /// SimCardSelector.GetSelectedCards for an Armaments pick) clobber each
    /// other's request_id forever: every agent answer arrives "stale", both
    /// sides retry, and the session livelocks at wire speed (2026-07-12
    /// seed-1005: 65k messages / 3600s timeout). One in-flight request at a
    /// time makes the second caller queue instead of clobber.
    /// </summary>
    private static readonly SemaphoreSlim RequestLock = new(1, 1);

    /// <summary>
    /// Blocks until a client is connected. Resets the AutoSlay watchdog
    /// periodically so a long wait for a human/agent isn't mistaken for a
    /// stuck automation loop.
    /// </summary>
    public static async Task WaitForClientAsync(CancellationToken ct, string context)
    {
        bool loggedWaiting = false;
        while (!BridgeServer.Instance.IsClientConnected)
        {
            ct.ThrowIfCancellationRequested();
            if (!loggedWaiting)
            {
                Logger.Log($"[SimAgentGate] No agent connected, waiting before proceeding with {context}...");
                loggedWaiting = true;
            }
            SimAutoSlayer.CurrentWatchdog?.Reset($"Waiting for agent connection ({context})");
            await Task.Delay(500, ct);
        }
    }

    /// <summary>
    /// Sends a state message and blocks until a correlated agent response
    /// arrives. Retries indefinitely (waiting for (re)connection as needed)
    /// rather than ever returning a fabricated/random choice.
    /// </summary>
    public static async Task<string> RequestActionAsync(string stateJson, CancellationToken ct, string context)
    {
        await RequestLock.WaitAsync(ct);
        // The whole time this gate holds, the game is only WAITING for the
        // agent -- render as little of the idle screen as possible so the
        // agent's search gets the machine (SimThinkFps.cs). Balanced in the
        // finally: every exit path (answer, cancel, throw) releases the cap.
        SimThinkFps.Begin();
        try
        {
            while (true)
            {
                ct.ThrowIfCancellationRequested();
                await WaitForClientAsync(ct, context);

                // Send ONCE, then heartbeat the watchdog while the agent
                // thinks. The reset below (on attempt expiry) only fires every
                // PerAttemptTimeout -- a day with agent_timeout_seconds=0 --
                // while SimCombatHandler's WaitHelper polls the SAME watchdog
                // during the wait. An MCTS agent thinking 30-120s per decision
                // (4000-sim boss searches) tripped it mid-combat: 6 of the
                // first 11 runs of the 2026-08-02 live sweep died as
                // "Watchdog timeout: No progress ... Combat turn N" including
                // a floor-48 run. Re-sending instead (shorter attempts) is not
                // an option: a duplicate state message would race the client's
                // reply to the first copy.
                Task<string?> waitTask = BridgeServer.Instance.SendStateAndWaitForActionAsync(
                    stateJson, PerAttemptTimeout, ct);
                // AgentHeartbeat.cs: the old inline loop spun on the main
                // thread once ct was cancelled (public issue #53).
                await AgentHeartbeat.WaitAsync(waitTask, TimeSpan.FromSeconds(5), ct,
                    () => SimAutoSlayer.CurrentWatchdog?.Reset($"Awaiting agent decision ({context})"));
                // WaitForPendingAsync cancels on ct too and never throws, so
                // this completes (with null) after a stop -- a real yield.
                string? response = await waitTask;
                // A stop, not a timeout: unwind instead of logging "No response".
                ct.ThrowIfCancellationRequested();

                if (response != null)
                {
                    return response;
                }

                Logger.Log($"[SimAgentGate] No response for {context} within {PerAttemptTimeout.TotalSeconds:0}s " +
                           $"(client_connected={BridgeServer.Instance.IsClientConnected}); retrying, not falling back.");
                SimAutoSlayer.CurrentWatchdog?.Reset($"Awaiting agent decision ({context})");
            }
        }
        finally
        {
            SimThinkFps.End();
            RequestLock.Release();
        }
    }
}
