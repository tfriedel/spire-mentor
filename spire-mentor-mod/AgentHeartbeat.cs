// AgentHeartbeat.cs -- waiting for the agent's reply with a periodic
// heartbeat, without spinning once the run is cancelled.
//
// Kept free of game types so tests/unit can drive it on a single-threaded
// SynchronizationContext, the way Godot runs it.
//
// The bug it fixes (public issue #53): SimAgentGate looped
//     while (true) { done = await Task.WhenAny(waitTask, Task.Delay(5000, ct)); if (done == waitTask) break; ... }
// and nothing in it checked ct. Once F8 cancelled the run, Task.Delay(.., ct)
// returned an already-cancelled task, WhenAny completed synchronously and the
// await continued inline, so the loop never yielded. waitTask only completes
// through a continuation queued on the same (Godot main-thread) context, so
// when the gate's continuation ran first the main thread spun forever: the
// game went Not Responding at 100% of one core. Live, 3 freezes in 7 F8
// presses with an agent that had not answered yet (card reward, combat turn,
// card select; logs/2026-11-26-issue53-live on the Windows box).

using System;
using System.Threading;
using System.Threading.Tasks;

namespace SpireMentor;

public static class AgentHeartbeat
{
    /// <summary>
    /// Returns once <paramref name="waitTask"/> completes or
    /// <paramref name="ct"/> is cancelled, calling <paramref name="onBeat"/>
    /// every <paramref name="interval"/> in between. Never throws on
    /// cancellation: the caller then awaits <paramref name="waitTask"/>,
    /// which really yields to the context, so the continuation that finishes
    /// it can run.
    /// </summary>
    public static async Task WaitAsync(Task waitTask, TimeSpan interval, CancellationToken ct, Action onBeat)
    {
        while (!waitTask.IsCompleted && !ct.IsCancellationRequested)
        {
            Task done = await Task.WhenAny(waitTask, Task.Delay(interval, ct));
            if (done == waitTask || ct.IsCancellationRequested)
                return;
            onBeat();
        }
    }
}
