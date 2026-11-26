// AgentHeartbeatTests.cs -- the agent wait must yield after a stop (public
// issue #53): on Godot's single main-thread context, a loop that never
// yields starves the continuation that would end it.

using System.Collections.Concurrent;
using SpireMentor;

internal static class AgentHeartbeatTests
{
    /// <summary>A one-thread context like Godot's: posts run in order on the pump thread.</summary>
    private sealed class PumpContext : SynchronizationContext
    {
        public readonly BlockingCollection<(SendOrPostCallback, object?)> Queue = new();
        public override void Post(SendOrPostCallback d, object? state) => Queue.Add((d, state));
    }

    public static void Run(Action<string, string?, string?> check)
    {
        check("heartbeat: a stop returns without the wait finishing", StopWhileSilent(), "unwound");
        check("heartbeat: a stop while the agent is silent unwinds the gate", StopGate(), "unwound");
        check("heartbeat: a reply ends the wait", ReplyArrives(), "reply");
        check("heartbeat: beats while waiting", BeatsWhileWaiting(), "True");
    }

    /// <summary>
    /// The live freeze: once the run is cancelled, the heartbeat must return
    /// on its own -- not only when the wait finishes, because in the game
    /// the wait's completion is a continuation queued behind the gate's on
    /// the same main-thread context. Here the wait never finishes at all.
    /// The old loop never noticed the cancel: every Task.Delay(.., ct) came
    /// back already cancelled, so it either spun inside one callback (what
    /// froze the game) or beat in a hot loop; neither ever returns.
    /// </summary>
    private static string StopWhileSilent()
    {
        var cts = new CancellationTokenSource();
        var never = new TaskCompletionSource<string?>(TaskCreationOptions.RunContinuationsAsynchronously);
        int beats = 0;
        return OnPump(ctx =>
        {
            Task heartbeat = AgentHeartbeat.WaitAsync(never.Task, TimeSpan.FromSeconds(5), cts.Token, () => beats++);
            ctx.Post(_ => cts.Cancel(), null);
            return heartbeat;
        }, hb => hb.IsCompletedSuccessfully && beats == 0 ? "unwound" : $"heartbeat {hb.Status}, {beats} beats");
    }

    /// <summary>The whole gate after a stop: heartbeat, await the (cancelled) wait, unwind.</summary>
    private static string StopGate()
    {
        var cts = new CancellationTokenSource();
        var reply = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        return OnPump(ctx =>
        {
            Task<string?> waitTask = PendingAsync(reply, cts.Token);
            Task gate = GateAsync(waitTask, cts.Token, TimeSpan.FromSeconds(5), () => { });
            ctx.Post(_ => cts.Cancel(), null);
            return gate;
        }, gate => gate.IsCanceled ? "unwound" : $"gate ended {gate.Status}");
    }

    private static string ReplyArrives()
    {
        var cts = new CancellationTokenSource();
        var reply = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<string?>? waitTask = null;
        return OnPump(ctx =>
        {
            waitTask = PendingAsync(reply, cts.Token);
            Task gate = GateAsync(waitTask, cts.Token, TimeSpan.FromSeconds(5), () => { });
            ctx.Post(_ => reply.TrySetResult("reply"), null);
            return gate;
        }, gate => gate.IsCompletedSuccessfully ? waitTask!.Result ?? "null" : $"gate ended {gate.Status}");
    }

    private static string BeatsWhileWaiting()
    {
        var cts = new CancellationTokenSource();
        var reply = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        int beats = 0;
        return OnPump(ctx =>
        {
            Task<string?> waitTask = PendingAsync(reply, cts.Token);
            Task gate = GateAsync(waitTask, cts.Token, TimeSpan.FromMilliseconds(20), () =>
            {
                if (++beats == 3)
                    reply.TrySetResult("reply");
            });
            return gate;
        }, gate => (gate.IsCompletedSuccessfully && beats == 3).ToString());
    }

    /// <summary>SimAgentGate's shape: heartbeat, await the wait, unwind on a stop.</summary>
    private static async Task GateAsync(Task<string?> waitTask, CancellationToken ct, TimeSpan interval, Action onBeat)
    {
        await AgentHeartbeat.WaitAsync(waitTask, interval, ct, onBeat);
        await waitTask;
        ct.ThrowIfCancellationRequested();
    }

    /// <summary>BridgeServer.WaitForPendingAsync's shape: null on cancellation, never throws.</summary>
    private static async Task<string?> PendingAsync(TaskCompletionSource<string> reply, CancellationToken ct)
    {
        using CancellationTokenRegistration reg = ct.Register(() => reply.TrySetCanceled());
        try
        {
            return await reply.Task;
        }
        catch (OperationCanceledException)
        {
            return null;
        }
    }

    /// <summary>
    /// Runs <paramref name="start"/> on a pump thread and pumps until its task
    /// ends. A pump stuck inside one callback (the spin) is reported after 5 s.
    /// </summary>
    private static string OnPump(Func<PumpContext, Task> start, Func<Task, string> result)
    {
        var ctx = new PumpContext();
        Task? task = null;
        var finished = new ManualResetEventSlim();
        var pump = new Thread(() =>
        {
            SynchronizationContext.SetSynchronizationContext(ctx);
            task = start(ctx);
            DateTime deadline = DateTime.UtcNow.AddSeconds(5);
            while (!task.IsCompleted && DateTime.UtcNow < deadline)
            {
                if (ctx.Queue.TryTake(out var item, 50))
                    item.Item1(item.Item2);
            }
            finished.Set();
        }) { IsBackground = true };
        pump.Start();
        if (!finished.Wait(TimeSpan.FromSeconds(10)))
            return "pump thread stuck (spinning)";
        return task!.IsCompleted ? result(task) : "timed out";
    }
}
