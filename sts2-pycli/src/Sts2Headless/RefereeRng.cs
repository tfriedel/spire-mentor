using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using HarmonyLib;
using MegaCrit.Sts2.Core.Random;

namespace Sts2Headless;

/// <summary>
/// The referee-mode switch. Kept apart from <see cref="RefereeRng"/> so that
/// reading it never loads a game type: Program.Main reads it before the
/// assembly resolver for the game DLLs has been exercised.
/// </summary>
internal static class Referee
{
    public static readonly bool Mode =
        Environment.GetEnvironmentVariable("STS2_REFEREE") == "1";

    /// <summary>
    /// Operator diagnostics (<c>referee_stats</c>, <c>referee_selftest</c>,
    /// <c>referee_project</c>). Off unless the process starts with
    /// <c>STS2_REFEREE_DIAG=1</c>: the self-test re-arms the generator with a
    /// key the caller picks and the stats count draws, so a worker that plays
    /// scored episodes must not accept them.
    /// </summary>
    public static readonly bool Diagnostics =
        Environment.GetEnvironmentVariable("STS2_REFEREE_DIAG") == "1";

    /// <summary>
    /// Trial.DoubleDown abandons the run, as in the game. Always on in referee
    /// mode; the separate switch lets a default host (which can force the
    /// event) test it.
    /// </summary>
    /// <summary>
    /// The Crystal Sphere minigame is played by the client, one divination per
    /// action, instead of by the host's fixed sweep. Same switch pattern.
    /// </summary>
    public static readonly bool InteractiveSphere =
        Mode || Environment.GetEnvironmentVariable("STS2_CRYSTAL_SPHERE_INTERACTIVE") == "1";

    public static readonly bool DoubleDownAbandons =
        Mode || Environment.GetEnvironmentVariable("STS2_TRIAL_DOUBLE_DOWN_ABANDONS") == "1";

    /// <summary>
    /// GH #1282: the reward screen is the player's. After a combat and in a
    /// treasure room the host lists every reward in a <c>rewards</c> decision
    /// (take or skip each, in any order, then proceed) instead of taking gold,
    /// relics, potions and special cards itself. Same switch pattern.
    /// </summary>
    public static readonly bool InteractiveRewards =
        Mode || Environment.GetEnvironmentVariable("STS2_INTERACTIVE_REWARDS") == "1";

    /// <summary>
    /// GH #1286: a pick the game opens at the start of the player's turn is
    /// the player's. Entropy's "transform a card in hand" (EntropyPower.
    /// AfterPlayerTurnStart) runs as the game wrote it and surfaces as a
    /// card_select, instead of the host's stand-in that transforms the first
    /// card in hand (kept by default: the Rust engine mirrors it for parity).
    /// Same switch pattern.
    /// </summary>
    public static readonly bool InteractiveTurnStartPicks =
        Mode || Environment.GetEnvironmentVariable("STS2_INTERACTIVE_TURN_START_PICKS") == "1";
}

/// <summary>
/// Referee mode: every gameplay draw comes from a keyed PRF instead of the
/// game's own generator, so a client that sees every outcome still cannot
/// predict the next one.
///
/// Why: the game seeds each stream from a 32-bit value (RunRngSet.cs:107) and
/// expands it with Xoshiro256** (MegaRandom.cs), which is linear. Both make
/// the stream recoverable from observed draws.
///
/// How: draw n of a stream is <c>HMAC-SHA256(streamKey, n)</c> with
/// <c>streamKey = HMAC-SHA256(episodeKey, stream seed)</c>. The stream seed is
/// only a public label here. The range mapping copies MegaRandom's, so the
/// distributions are the game's.
///
/// The patches skip the original methods, so a patched stream's MegaRandom
/// never advances. <see cref="Audit"/> checks exactly that and is the proof
/// that no draw bypassed the patches (for example through JIT inlining).
///
/// Off unless the process starts with <c>STS2_REFEREE=1</c>; without it no
/// patch is installed and the parity path is untouched.
/// </summary>
internal static class RefereeRng
{
#if GAME_V0111
    // v0.111.0 changed the Rng shape (ulong seeds, no Seed / Counter
    // properties). Referee mode is not ported to it yet.
    public static void Install() =>
        throw new NotSupportedException("referee mode is not ported to v0.111.0");
    public static void Arm(byte[] episodeKey) => Install();
    public static string? Audit() => null;
    public static Dictionary<string, object?> Stats() => new();
    public static Dictionary<string, object?> SelfTest(int samples) => new();
#else
    private sealed class Stream
    {
        public byte[]? Key;
        public ulong S0, S1, S2, S3;
    }

    private static readonly object Gate = new();
    private static readonly ConditionalWeakTable<Rng, Stream> Streams = new();
    private static readonly List<WeakReference<Rng>> Tracked = new();
    private static byte[]? _episodeKey;
    private static bool _installed;
    private static long _draws;

    private static readonly AccessTools.FieldRef<Rng, int> CounterRef =
        AccessTools.FieldRefAccess<Rng, int>("<Counter>k__BackingField");
    private static readonly AccessTools.FieldRef<Rng, MegaRandom> RandomRef =
        AccessTools.FieldRefAccess<Rng, MegaRandom>("_random");
    private static readonly AccessTools.FieldRef<MegaRandom, ulong> S0Ref =
        AccessTools.FieldRefAccess<MegaRandom, ulong>("_s0");
    private static readonly AccessTools.FieldRef<MegaRandom, ulong> S1Ref =
        AccessTools.FieldRefAccess<MegaRandom, ulong>("_s1");
    private static readonly AccessTools.FieldRef<MegaRandom, ulong> S2Ref =
        AccessTools.FieldRefAccess<MegaRandom, ulong>("_s2");
    private static readonly AccessTools.FieldRef<MegaRandom, ulong> S3Ref =
        AccessTools.FieldRefAccess<MegaRandom, ulong>("_s3");

    private static bool Armed => _episodeKey != null;

    /// <summary>Installs the patches. Call before any game code runs.</summary>
    public static void Install()
    {
        if (_installed) return;
        _installed = true;

        var harmony = new Harmony("sts2headless.referee.rng");
        var rng = typeof(Rng);
        var self = typeof(RefereeRng);

        // Negative control for Audit(): STS2_REFEREE_SELFTEST_SKIP=<method>
        // leaves that method unpatched, so its draws reach the game generator
        // and the audit must report them. Test-only; a host started this way
        // faults on the first such draw.
        var skip = Environment.GetEnvironmentVariable("STS2_REFEREE_SELFTEST_SKIP");

        void Prefix(string method, Type[] sig, string patch)
        {
            if (method == skip) return;
            var target = rng.GetMethod(method, sig)
                ?? throw new MissingMethodException(rng.FullName, method);
            harmony.Patch(target, prefix: new HarmonyMethod(self.GetMethod(patch, AccessTools.all)));
        }

        // Every Rng constructor chains into this one.
        var ctor = rng.GetConstructor(new[] { typeof(uint), typeof(int) })
            ?? throw new MissingMethodException(rng.FullName, ".ctor(uint, int)");
        harmony.Patch(ctor, postfix: new HarmonyMethod(self.GetMethod(nameof(CtorPostfix), AccessTools.all)));

        // The complete set of Rng methods that touch the generator. Everything
        // else on Rng (NextItem, Shuffle, the Gaussians, ...) calls these.
        Prefix("FastForwardCounter", new[] { typeof(int) }, nameof(FastForwardPrefix));
        Prefix("NextBool", Type.EmptyTypes, nameof(NextBoolPrefix));
        Prefix("NextInt", new[] { typeof(int) }, nameof(NextInt1Prefix));
        Prefix("NextInt", new[] { typeof(int), typeof(int) }, nameof(NextInt2Prefix));
        Prefix("NextUnsignedInt", new[] { typeof(uint), typeof(uint) }, nameof(NextUInt2Prefix));
        Prefix("NextFloat", new[] { typeof(float), typeof(float) }, nameof(NextFloat2Prefix));
        Prefix("NextDouble", Type.EmptyTypes, nameof(NextDouble0Prefix));
        Prefix("NextDouble", new[] { typeof(double), typeof(double) }, nameof(NextDouble2Prefix));

        // Rng.Chaotic is seeded from the wall clock. Track it like any other
        // stream (its static initialiser runs here, after the ctor patch).
        Track(Rng.Chaotic);
    }

    /// <summary>
    /// Sets the episode key. From here on every draw is keyed and the game's
    /// generators must stand still.
    /// </summary>
    public static void Arm(byte[] episodeKey)
    {
        if (!_installed)
            throw new InvalidOperationException("referee patches are not installed");
        if (episodeKey.Length != 32)
            throw new ArgumentException("episode key must be 32 bytes");
        lock (Gate)
        {
            if (Armed)
                throw new InvalidOperationException("referee key is already set");
            // Streams built before the key existed may have drawn from the
            // game's generator; their baseline is the state as of now.
            foreach (var weak in Tracked)
            {
                if (weak.TryGetTarget(out var rng) && Streams.TryGetValue(rng, out var stream))
                    Snapshot(rng, stream);
            }
            _episodeKey = (byte[])episodeKey.Clone();
        }
    }

    /// <summary>
    /// Returns null when every tracked stream's game generator is untouched
    /// since it was keyed, else a description of the first stream that moved.
    /// </summary>
    public static string? Audit()
    {
        lock (Gate)
        {
            if (!Armed) return null;
            for (int i = Tracked.Count - 1; i >= 0; i--)
            {
                if (!Tracked[i].TryGetTarget(out var rng))
                {
                    Tracked.RemoveAt(i);
                    continue;
                }
                if (!Streams.TryGetValue(rng, out var stream)) continue;
                var mega = RandomRef(rng);
                if (S0Ref(mega) != stream.S0 || S1Ref(mega) != stream.S1 ||
                    S2Ref(mega) != stream.S2 || S3Ref(mega) != stream.S3)
                {
                    return $"rng escape: stream seed={rng.Seed} counter={rng.Counter} drew from the game generator";
                }
            }
            return null;
        }
    }

    public static Dictionary<string, object?> Stats()
    {
        lock (Gate)
        {
            int live = 0;
            foreach (var weak in Tracked)
                if (weak.TryGetTarget(out _)) live++;
            return new Dictionary<string, object?>
            {
                ["type"] = "referee_stats",
                ["armed"] = Armed,
                ["draws"] = _draws,
                ["live_streams"] = live,
            };
        }
    }

    /// <summary>
    /// Draws through the game's public Rng API on a fresh stream and returns
    /// histograms, so a test can check the keyed path is uniform end to end
    /// (range mapping, Shuffle) without playing a run.
    /// </summary>
    public static Dictionary<string, object?> SelfTest(int samples)
    {
        var rng = new Rng(0xC0FFEEu);
        var ints = new int[10];
        var bools = new int[2];
        var floats = new int[10];
        var firstAfterShuffle = new int[5];
        for (int i = 0; i < samples; i++)
        {
            ints[rng.NextInt(10)]++;
            bools[rng.NextBool() ? 1 : 0]++;
            floats[Math.Min(9, (int)(rng.NextFloat(1f) * 10f))]++;
            var deck = new List<int> { 0, 1, 2, 3, 4 };
            rng.Shuffle(deck);
            firstAfterShuffle[deck[0]]++;
        }
        return new Dictionary<string, object?>
        {
            ["type"] = "referee_selftest",
            ["samples"] = samples,
            ["counter"] = rng.Counter,
            ["next_int_10"] = ints,
            ["next_bool"] = bools,
            ["next_float_decile"] = floats,
            ["shuffle5_first"] = firstAfterShuffle,
        };
    }

    private static Stream Track(Rng rng)
    {
        lock (Gate)
        {
            if (Streams.TryGetValue(rng, out var known)) return known;
            var stream = new Stream();
            Snapshot(rng, stream);
            Streams.Add(rng, stream);
            Tracked.Add(new WeakReference<Rng>(rng));
            return stream;
        }
    }

    private static void Snapshot(Rng rng, Stream stream)
    {
        var mega = RandomRef(rng);
        stream.S0 = S0Ref(mega);
        stream.S1 = S1Ref(mega);
        stream.S2 = S2Ref(mega);
        stream.S3 = S3Ref(mega);
    }

    /// <summary>
    /// Advances the stream's counter and returns the keyed draw as a double in
    /// [0, 1), built exactly like MegaRandom.NextDouble (53 bits).
    /// </summary>
    private static double Draw(Rng rng)
    {
        var stream = Track(rng);
        byte[] key;
        lock (Gate)
        {
            key = stream.Key ??= DeriveStreamKey(rng.Seed);
            _draws++;
        }
        int counter = ++CounterRef(rng);

        Span<byte> message = stackalloc byte[8];
        BitConverter.TryWriteBytes(message, (long)counter);
        Span<byte> mac = stackalloc byte[32];
        HMACSHA256.HashData(key, message, mac);
        ulong raw = BitConverter.ToUInt64(mac);
        return (double)(raw >> 11) * 1.1102230246251565E-16;
    }

    private static byte[] DeriveStreamKey(uint seed)
    {
        Span<byte> label = stackalloc byte[4];
        BitConverter.TryWriteBytes(label, seed);
        return HMACSHA256.HashData(_episodeKey!, label);
    }

    // MegaRandom.Next(int): (int)(NextDouble() * max), max >= 1.
    private static int ToInt(Rng rng, int maxExclusive)
    {
        if (maxExclusive < 1)
            throw new ArgumentOutOfRangeException("maxValue", maxExclusive, "maxValue must be > 0");
        return (int)(Draw(rng) * (double)maxExclusive);
    }

    private static void CtorPostfix(Rng __instance) => Track(__instance);

    private static bool FastForwardPrefix(Rng __instance, int targetCount)
    {
        if (!Armed) return true;
        if (__instance.Counter > targetCount)
            throw new InvalidOperationException(
                $"Cannot fast-forward an Rng counter to a lower number (current = {__instance.Counter}, target = {targetCount})");
        CounterRef(__instance) = targetCount;
        return false;
    }

    private static bool NextBoolPrefix(Rng __instance, ref bool __result)
    {
        if (!Armed) return true;
        __result = ToInt(__instance, 2) == 0;
        return false;
    }

    private static bool NextInt1Prefix(Rng __instance, ref int __result, int maxExclusive)
    {
        if (!Armed) return true;
        __result = ToInt(__instance, maxExclusive);
        return false;
    }

    private static bool NextInt2Prefix(Rng __instance, ref int __result, int minInclusive, int maxExclusive)
    {
        if (!Armed) return true;
        if (minInclusive >= maxExclusive)
            throw new ArgumentOutOfRangeException(nameof(minInclusive), "Minimum must be lower than maximum.");
        // MegaRandom.Next(int, int).
        long range = (long)maxExclusive - minInclusive;
        __result = range <= int.MaxValue
            ? (int)(Draw(__instance) * (double)(int)range) + minInclusive
            : (int)((long)(Draw(__instance) * (double)range) + minInclusive);
        return false;
    }

    private static bool NextUInt2Prefix(Rng __instance, ref uint __result, uint minInclusive, uint maxExclusive)
    {
        if (!Armed) return true;
        if (minInclusive >= maxExclusive)
            throw new ArgumentOutOfRangeException(nameof(minInclusive), "Minimum must be lower than maximum.");
        double span = maxExclusive - minInclusive;
        __result = minInclusive + (uint)(Draw(__instance) * span);
        return false;
    }

    private static bool NextFloat2Prefix(Rng __instance, ref float __result, float min, float max)
    {
        if (!Armed) return true;
        if (min > max)
            throw new ArgumentOutOfRangeException(nameof(min), "Minimum must not be higher than maximum.");
        __result = (float)(Draw(__instance) * (double)(max - min) + (double)min);
        return false;
    }

    private static bool NextDouble0Prefix(Rng __instance, ref double __result)
    {
        if (!Armed) return true;
        __result = Draw(__instance);
        return false;
    }

    private static bool NextDouble2Prefix(Rng __instance, ref double __result, double min, double max)
    {
        if (!Armed) return true;
        if (min > max)
            throw new ArgumentOutOfRangeException(nameof(min), "Minimum must not be higher than maximum.");
        __result = Draw(__instance) * (max - min) + min;
        return false;
    }
#endif
}
