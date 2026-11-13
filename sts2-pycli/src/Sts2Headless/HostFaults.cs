using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using HarmonyLib;

namespace Sts2Headless;

/// <summary>
/// GH #1283: every place where a game error is swallowed and the run carries
/// on, counted per site.
///
/// Three sources feed it:
/// <list type="bullet">
/// <item>the host's own empty <c>catch</c> blocks (<see cref="Swallowed"/>,
///   site <c>catch:Member:line</c>);</item>
/// <item>the host's <c>LogException</c> and event force-finish paths
///   (<see cref="Record"/>, sites <c>log:context</c> and
///   <c>force-finish:where</c>);</item>
/// <item>the game's own swallows: <c>TaskHelper.RunSafely</c>, the action
///   queue and the rest all report through
///   <c>SentryService.CaptureException</c>, which is hooked here (site
///   <c>game:Caller</c>).</item>
/// </list>
///
/// In default mode this only counts (the <c>host_faults</c> command reads
/// the census). In referee mode the first fault of a command ends the episode
/// with <c>referee_fault</c> (Program.cs): a run that went through a dropped
/// effect is void, not scored.
/// </summary>
internal static class HostFaults
{
    private static readonly object Gate = new();
    private static readonly Dictionary<string, int> Counts = new();
    private static readonly Dictionary<string, string> Samples = new();
    private static string? _pending;

    /// <summary>Operator logging of each fault to stderr (census runs).</summary>
    private static readonly bool LogToStderr =
        Environment.GetEnvironmentVariable("STS2_HOST_FAULT_LOG") == "1";

    /// <summary>
    /// Test-only: names one site (as <see cref="Swallowed"/> spells it) whose
    /// try block throws before it runs, so the fail-closed path can be tested
    /// on a real site. Unset in every real run.
    /// </summary>
    private static readonly string? Inject =
        Environment.GetEnvironmentVariable("STS2_HOST_FAULT_INJECT");

    /// <summary>An empty <c>catch</c> in the host swallowed <paramref name="ex"/>.</summary>
    public static void Swallowed(Exception ex,
        [CallerMemberName] string member = "", [CallerLineNumber] int line = 0) =>
        Record($"catch:{member}:{line}", ex);

    /// <summary>
    /// Throws when <see cref="Inject"/> names this site. Called first thing in
    /// the try block of the site the fail-closed test forces.
    /// </summary>
    public static void MaybeInject(string site)
    {
        if (Inject != null && Inject == site)
            throw new InvalidOperationException($"injected fault at {site}");
    }

    public static void Record(string site, Exception? ex)
    {
        string what = ex == null ? "" : Describe(ex);
        lock (Gate)
        {
            Counts[site] = Counts.GetValueOrDefault(site) + 1;
            Samples.TryAdd(site, what);
            _pending ??= what.Length == 0 ? site : $"{site} ({what})";
        }
        // Not on stderr by default: an exception type name there scores as a
        // world error in the Python client, and default mode must not change.
        if (LogToStderr) Console.Error.WriteLine($"[HOST_FAULT] {site} {what}");
    }

    private static string Describe(Exception ex)
    {
        while (ex.InnerException != null) ex = ex.InnerException;
        string message = ex.Message.Length > 160 ? ex.Message[..160] : ex.Message;
        return $"{ex.GetType().Name}: {message}";
    }

    /// <summary>The first fault since the last call, or null.</summary>
    public static string? TakePending()
    {
        lock (Gate)
        {
            var pending = _pending;
            _pending = null;
            return pending;
        }
    }

    /// <summary>Per-site counts and one sample each, for the census.</summary>
    public static Dictionary<string, object?> Census()
    {
        lock (Gate)
        {
            return new Dictionary<string, object?>
            {
                ["type"] = "host_faults",
                ["counts"] = Counts.OrderBy(kv => kv.Key, StringComparer.Ordinal)
                    .ToDictionary(kv => kv.Key, kv => (object?)kv.Value),
                ["samples"] = Samples.OrderBy(kv => kv.Key, StringComparer.Ordinal)
                    .ToDictionary(kv => kv.Key, kv => (object?)kv.Value),
            };
        }
    }

    /// <summary>
    /// Hooks SentryService.CaptureException, the one call the game makes
    /// wherever it swallows an exception and goes on.
    /// </summary>
    public static void InstallGameHooks()
    {
        try
        {
            var harmony = new Harmony("sts2headless.host-faults");
            var sentry = typeof(MegaCrit.Sts2.Core.Debug.SentryService);
            var prefix = new HarmonyMethod(typeof(HostFaults).GetMethod(nameof(CapturePrefix), AccessTools.all));
            int patched = 0;
            foreach (var method in sentry.GetMethods(AccessTools.all))
            {
                if (method.Name != "CaptureException") continue;
                var first = method.GetParameters().FirstOrDefault();
                if (first == null || first.ParameterType != typeof(Exception)) continue;
                harmony.Patch(method, prefix: prefix);
                patched++;
            }
            Console.Error.WriteLine($"[INFO] HostFaults: hooked {patched} Sentry capture overload(s)");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[WARN] HostFaults: could not hook SentryService: {ex.Message}");
        }
    }

    private static void CapturePrefix(Exception __0)
    {
        // The game frame that reported it: the first one outside SentryService.
        string caller = "unknown";
        var frames = new System.Diagnostics.StackTrace(1, false).GetFrames();
        foreach (var frame in frames)
        {
            var type = frame.GetMethod()?.DeclaringType;
            if (type == null || type == typeof(HostFaults) || type.Name.Contains("SentryService")) continue;
            // Async state machines are nested types named "<Method>d__N".
            var owner = type.DeclaringType != null && type.Name.StartsWith('<') ? type.DeclaringType : type;
            string method = type.Name.StartsWith('<') ? type.Name[1..type.Name.IndexOf('>')] : frame.GetMethod()!.Name;
            caller = $"{owner.Name}.{method}";
            break;
        }
        Record($"game:{caller}", __0);
    }
}
