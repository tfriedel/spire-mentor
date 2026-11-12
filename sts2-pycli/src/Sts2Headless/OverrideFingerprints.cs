using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using HarmonyLib;

namespace Sts2Headless;

/// <summary>
/// Fingerprints every game method whose original body the host can skip (GH #1299).
///
/// A Harmony prefix that returns <c>bool</c> can stop the original from running. The
/// host then runs something else in its place: a copy of the rules written from the
/// decomp, or nothing, on the claim that the method is presentation-only. Either way
/// the host keeps the rules of the build the patch was written against. When a game
/// update changes such a method, its fingerprint changes, and
/// <c>tests/test_host_override_fingerprints.py</c> fails until somebody re-reads the
/// new original and confirms the override still holds.
///
/// The fingerprint hashes the original IL with every operand resolved to a name, so
/// unrelated metadata-token renumbering between builds does not change it. For an
/// async or iterator method the logic lives in the compiler-generated state machine,
/// so its <c>MoveNext</c> is hashed as well. Calls into the method's own class and its
/// nested types (private helpers, lambdas, local functions, display classes) are
/// followed transitively, since the rules often live there: RollingBoulderPower's
/// damage is in its private <c>DoDamage</c>. Calls to other classes are not followed.
///
/// Only methods declared in the game assembly are listed; the BCL patches (for
/// example <c>YieldAwaiter.IsCompleted</c>) change with the .NET runtime, not the game.
/// </summary>
internal static class OverrideFingerprints
{
    public static Dictionary<string, object?> Collect(bool dump)
    {
        var gameAssembly = typeof(MegaCrit.Sts2.Core.Commands.Cmd).Assembly;
        var rows = new List<Dictionary<string, object?>>();
        foreach (var method in Harmony.GetAllPatchedMethods())
        {
            if (method.DeclaringType?.Assembly != gameAssembly) continue;
            var info = Harmony.GetPatchInfo(method);
            if (info == null) continue;
            var skipping = info.Prefixes
                .Where(p => p.PatchMethod.ReturnType == typeof(bool))
                .Select(p => $"{p.PatchMethod.DeclaringType?.Name}.{p.PatchMethod.Name}")
                .OrderBy(n => n, StringComparer.Ordinal)
                .ToList();
            if (skipping.Count == 0) continue;

            var text = RenderClosure(method);

            var row = new Dictionary<string, object?>
            {
                ["method"] = Key(method),
                ["prefixes"] = skipping,
                ["hash"] = Hash(text),
            };
            if (dump) row["il"] = text;
            rows.Add(row);
        }
        rows.Sort((a, b) => string.CompareOrdinal((string)a["method"]!, (string)b["method"]!));
        return new Dictionary<string, object?>
        {
            ["type"] = "override_fingerprints",
            ["overrides"] = rows,
        };
    }

    internal static string Key(MethodBase m)
    {
        var ps = string.Join(",", m.GetParameters().Select(p => p.ParameterType.ToString()));
        return $"{m.DeclaringType}::{m.Name}({ps})";
    }

    /// <summary>The method, then every method it reaches inside its own class, each
    /// followed by its state machine's MoveNext, in first-reached order.</summary>
    private static string RenderClosure(MethodBase root)
    {
        var home = root.DeclaringType!;
        var seen = new HashSet<MethodBase>();
        var queue = new Queue<MethodBase>();
        var sb = new StringBuilder();
        void Enqueue(MethodBase m)
        {
            if (seen.Add(m)) queue.Enqueue(m);
        }
        Enqueue(root);
        while (queue.Count > 0)
        {
            var m = queue.Dequeue();
            if (m.GetMethodBody() == null) continue;
            sb.Append("--- ").Append(Key(m)).Append('\n');
            List<CodeInstruction> code;
            try
            {
                code = PatchProcessor.GetOriginalInstructions(m);
            }
            catch (Exception ex) when (ex is MissingMemberException or TypeLoadException)
            {
                // A presentation helper that names a Godot member the GodotStubs
                // lack cannot be decoded. Hash its raw IL instead: still changes
                // with the code, though also with metadata-token renumbering.
                sb.Append("raw-il ").Append(Convert.ToHexString(m.GetMethodBody()!.GetILAsByteArray()!)).Append('\n');
                continue;
            }
            sb.Append(Render(code));
            var sm = StateMachineMoveNext(m);
            if (sm != null) Enqueue(sm);
            foreach (var ci in code)
                if (ci.operand is MethodBase callee && callee != m && IsInside(callee.DeclaringType, home))
                    Enqueue(callee);
        }
        return sb.ToString();
    }

    private static bool IsInside(Type? t, Type home)
    {
        for (; t != null; t = t.DeclaringType)
            if (t == home) return true;
        return false;
    }

    private static MethodInfo? StateMachineMoveNext(MethodBase m)
    {
        var smType = m.GetCustomAttribute<AsyncStateMachineAttribute>()?.StateMachineType
                     ?? m.GetCustomAttribute<IteratorStateMachineAttribute>()?.StateMachineType;
        return smType?.GetMethod("MoveNext", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
    }

    private static string Hash(string text)
    {
        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(text));
        return Convert.ToHexString(bytes, 0, 8).ToLowerInvariant();
    }

    private static string Render(List<CodeInstruction> code)
    {
        var labelAt = new Dictionary<Label, int>();
        for (int i = 0; i < code.Count; i++)
            foreach (var l in code[i].labels)
                labelAt[l] = i;

        var sb = new StringBuilder();
        foreach (var ci in code)
        {
            sb.Append(ci.opcode.Name);
            var op = Operand(ci.operand, labelAt);
            if (op.Length > 0) sb.Append(' ').Append(op);
            foreach (var block in ci.blocks)
                sb.Append(" [").Append(block.blockType).Append(' ').Append(block.catchType).Append(']');
            sb.Append('\n');
        }
        return sb.ToString();
    }

    private static string Operand(object? operand, Dictionary<Label, int> labelAt) => operand switch
    {
        null => "",
        Label l => labelAt.TryGetValue(l, out var i) ? $"@{i}" : "@?",
        Label[] ls => string.Join(",", ls.Select(l => labelAt.TryGetValue(l, out var i) ? $"@{i}" : "@?")),
        LocalBuilder lb => $"loc{lb.LocalIndex}:{lb.LocalType}",
        MethodBase mb => Key(mb),
        FieldInfo f => $"{f.DeclaringType}::{f.Name}:{f.FieldType}",
        Type t => t.ToString(),
        string s => "\"" + s + "\"",
        IFormattable fmt => fmt.ToString(null, CultureInfo.InvariantCulture),
        _ => operand.ToString() ?? "",
    };
}
