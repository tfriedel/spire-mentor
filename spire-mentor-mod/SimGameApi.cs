// SimGameApi.cs -- the sts2 members whose signature differs between the game
// builds the mod supports, reached at runtime so ONE DLL runs on both.
//
// The mod compiles against whichever sts2.dll is installed, but a direct call
// binds the exact IL signature of that build and throws
// MissingMethodException on the other (0.7.0 on the Workshop was built
// against the beta and gave no advice on the stable game). Every member below
// is looked up by name instead; tests/compile/build_mod.sh runs RefCheck
// against every game build present so a new direct reference to a drifted
// member fails the gate.
//
//   member                            v0.107.1                    v0.111.0
//   RunRngSet.Seed                    uint                        ulong
//   Rng read count                    int Counter { get; }        private int _counter
//   new Rng(seed)                     Rng(uint, int = 0)          Rng(ulong = 0)
//   StringHelper.GetDeterministic-    int                         ulong
//     HashCode(string)
//   CardSelectCmd.UseSelector /       (ICardSelector)             (ICardSelector,
//     PushSelector                                                 bool localOnly = false)
//
// v0.113.0 (beta) moved more:
//
//   member                            v0.107.1 / v0.111.0         v0.113.0
//   card selectors                    static CardSelectCmd stack  per-run CardSelectorStack
//                                                                 (RunManager.CardSelectorStack)
//   MerchantEntry purchase            OnTryPurchaseWrapper(inv)   MerchantSynchronizer
//                                                                 .PurchaseLocalEntry(entry)
//   NHotkeyManager.PushHotkey-        (string, Action)            (string, Action,
//     PressedBinding                                               long sequence = 0)
//   new Watchdog()                    ()                          (Action<string> info,
//                                                                  Action<string> error, ...)
//   the shared watchdog               AutoSlayer.CurrentWatchdog  Watchdog.Current
//
// Sources: decomp v0.107.1 Core.Random/Rng.cs, Core.Runs/RunRngSet.cs,
// Core.Commands/CardSelectCmd.cs; the same files in v0.111.0; v0.113.0
// Core.CardSelection/CardSelectorStack.cs, Core.Multiplayer.Game/
// MerchantSynchronizer.cs, Core.Nodes.CommonUi/NHotkeyManager.cs:117,
// Core.AutoSlay.Helpers/Watchdog.cs:50-62.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.AutoSlay;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;

namespace SpireMentor;

internal static class SimGameApi
{
    private const BindingFlags Instance = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
    private const BindingFlags Static = BindingFlags.Static | BindingFlags.Public;

    private static readonly PropertyInfo RunSeedProperty =
        typeof(RunRngSet).GetProperty(nameof(RunRngSet.Seed), Instance)
        ?? throw Missing("RunRngSet.Seed");

    // v0.107.1 exposes the count as a property; v0.111.0 keeps only the field
    // (its public face is ToSerializable().counter, which allocates per call).
    private static readonly PropertyInfo? CounterProperty =
        typeof(Rng).GetProperty("Counter", Instance);
    private static readonly FieldInfo? CounterField =
        CounterProperty == null ? typeof(Rng).GetField("_counter", Instance) : null;

    private static readonly MethodInfo HashMethod =
        typeof(StringHelper).GetMethod(nameof(StringHelper.GetDeterministicHashCode), Static, new[] { typeof(string) })
        ?? throw Missing("StringHelper.GetDeterministicHashCode(string)");

    // The seed constructor: first parameter uint (v0.107.1) or ulong
    // (v0.111.0), every other parameter optional.
    private static readonly ConstructorInfo SeedConstructor =
        typeof(Rng).GetConstructors()
            .FirstOrDefault(c =>
            {
                ParameterInfo[] p = c.GetParameters();
                return p.Length >= 1
                    && (p[0].ParameterType == typeof(uint) || p[0].ParameterType == typeof(ulong))
                    && p.Skip(1).All(x => x.IsOptional);
            })
        ?? throw Missing("Rng(uint|ulong seed)");

    // Null on v0.113.0+, which has no static selector stack (see the
    // selector section below).
    private static readonly MethodInfo? UseSelectorMethod = SelectorMethod("UseSelector");
    private static readonly MethodInfo? PushSelectorMethod = SelectorMethod("PushSelector");

    /// <summary>
    /// The run's once-hashed master seed, boxed as the running build declares
    /// it (<c>uint</c> / <c>ulong</c>); both serialize as a JSON number.
    /// </summary>
    public static object RunSeed(RunRngSet rng) => RunSeedProperty.GetValue(rng)!;

    /// <summary>How many values the stream has produced.</summary>
    public static int Counter(Rng rng) =>
        (int)(CounterProperty != null ? CounterProperty.GetValue(rng)! : CounterField!.GetValue(rng)!);

    /// <summary>
    /// <c>new Rng(seed)</c>, the seed narrowed or widened to the build's type:
    /// v0.107.1 takes the uint truncation of its int hash, v0.111.0 the whole
    /// ulong.
    /// </summary>
    public static Rng NewRng(ulong seed)
    {
        ParameterInfo[] p = SeedConstructor.GetParameters();
        var args = new object?[p.Length];
        args[0] = p[0].ParameterType == typeof(uint) ? (object)unchecked((uint)seed) : seed;
        for (int i = 1; i < p.Length; i++) args[i] = p[i].DefaultValue;
        return (Rng)SeedConstructor.Invoke(args);
    }

    /// <summary>
    /// An Rng seeded the way the game's own AutoSlayer seeds one from a seed
    /// string: v0.107.1 truncates the int hash to uint (AutoSlayer.cs:167),
    /// v0.111.0 passes its ulong hash whole (AutoSlayer.cs:184).
    /// </summary>
    public static Rng RngFromSeedString(string seed)
    {
        object hash = HashMethod.Invoke(null, new object[] { seed })!;
        return NewRng(hash is int i ? unchecked((uint)i) : (ulong)hash);
    }

    /// <summary>
    /// <c>CardSelectCmd.UseSelector(selector)</c> on the shared stack: throws
    /// if a selector is active; disposing clears every selector.
    /// </summary>
    public static IDisposable UseSelector(ICardSelector selector)
    {
        if (UseSelectorMethod != null)
            return InvokeSelector(UseSelectorMethod, selector);
        if (OwnSelectors.Count > 0)
            throw new InvalidOperationException("A card selector is already active.");
        OwnSelectors.Add(selector);
        return new Scope(() => OwnSelectors.Clear());
    }

    /// <summary><c>CardSelectCmd.PushSelector(selector)</c> on the shared stack.</summary>
    public static IDisposable PushSelector(ICardSelector selector)
    {
        if (PushSelectorMethod != null)
            return InvokeSelector(PushSelectorMethod, selector);
        OwnSelectors.Add(selector);
        return new Scope(() => OwnSelectors.Remove(selector));
    }

    // ── Card selectors on v0.113.0+ ─────────────────────────────────────
    //
    // v0.113.0 keeps selectors on a CardSelectorStack that RunManager builds
    // per run (RunManager.cs:540), so one installed before the run (the
    // AutoSlayer does) would be lost, and an instance stack cannot be reached
    // from the menu at all. The mod keeps its own stack instead and serves it
    // from two postfixes, which reproduce the old static stacks exactly:
    //
    // * GetSelector(player) — the old shared stack. CardSelectCmd uses its
    //   answer for EVERY player's choice and skips PlayerChoiceSynchronizer,
    //   so it is served in single player only.
    // * LocalSelector — the old localOnly stack. Consulted only for the local
    //   player's choice, whose pick is then synced (CardSelectCmd.cs:92), so
    //   it is the co-op path.
    //
    // Both postfixes only fill a null answer: a selector the game itself
    // pushed (Whispering Earring, a test selector) still wins.

    private static readonly List<ICardSelector> OwnSelectors = new();

    /// <summary>
    /// Patch the v0.113.0+ CardSelectorStack to serve the mod's selectors.
    /// No-op on builds with the static CardSelectCmd stack. Call once at
    /// startup.
    /// </summary>
    public static void InstallSelectorBridge(Harmony harmony)
    {
        if (UseSelectorMethod != null)
            return;
        Type? stack = AccessTools.TypeByName("MegaCrit.Sts2.Core.CardSelection.CardSelectorStack");
        MethodInfo? getSelector = stack == null ? null : AccessTools.Method(stack, "GetSelector");
        MethodInfo? localSelector = stack == null ? null : AccessTools.PropertyGetter(stack, "LocalSelector");
        if (getSelector == null || localSelector == null)
            throw Missing("CardSelectorStack.GetSelector / LocalSelector");
        harmony.Patch(getSelector, postfix: new HarmonyMethod(typeof(SimGameApi), nameof(SharedSelectorPostfix)));
        harmony.Patch(localSelector, postfix: new HarmonyMethod(typeof(SimGameApi), nameof(LocalSelectorPostfix)));
        Logger.Log("  Patched: CardSelectorStack selector bridge (v0.113.0+)");
    }

    private static void SharedSelectorPostfix(ref ICardSelector? __result)
    {
        if (__result == null && OwnSelectors.Count > 0 && !SimMp.IsMultiplayer)
            __result = OwnSelectors[^1];
    }

    private static void LocalSelectorPostfix(ref ICardSelector? __result)
    {
        if (__result == null && OwnSelectors.Count > 0 && SimMp.IsMultiplayer)
            __result = OwnSelectors[^1];
    }

    private sealed class Scope : IDisposable
    {
        private Action? _onDispose;

        public Scope(Action onDispose) => _onDispose = onDispose;

        public void Dispose()
        {
            _onDispose?.Invoke();
            _onDispose = null;
        }
    }

    // ── Merchant ────────────────────────────────────────────────────────

    private static readonly MethodInfo? PurchaseWrapperMethod =
        typeof(MerchantEntry).GetMethod("OnTryPurchaseWrapper", Instance);

    /// <summary>
    /// Buy <paramref name="entry"/> from <paramref name="inventory"/> the way
    /// the shop UI does: v0.113.0 sends it through
    /// <c>RunManager.MerchantSynchronizer.PurchaseLocalEntry(entry)</c>
    /// (NMerchantSlot.cs:402; the synchronizer already holds the inventory,
    /// a FakeMerchant's included), older builds call
    /// <c>entry.OnTryPurchaseWrapper(inventory)</c>.
    /// </summary>
    public static Task<bool> Purchase(MerchantEntry entry, MerchantInventory? inventory)
    {
        if (PurchaseWrapperMethod != null)
            return (Task<bool>)Invoke(PurchaseWrapperMethod, entry, inventory, false)!;
        object runManager = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
        object synchronizer = AccessTools.Property(runManager.GetType(), "MerchantSynchronizer")?.GetValue(runManager)
            ?? throw Missing("RunManager.MerchantSynchronizer");
        MethodInfo purchase = AccessTools.Method(synchronizer.GetType(), "PurchaseLocalEntry")
            ?? throw Missing("MerchantSynchronizer.PurchaseLocalEntry");
        return (Task<bool>)Invoke(purchase, synchronizer, entry)!;
    }

    // ── Hotkeys ─────────────────────────────────────────────────────────

    private static readonly MethodInfo PushHotkeyMethod =
        typeof(NHotkeyManager).GetMethods(Instance)
            .FirstOrDefault(m =>
            {
                ParameterInfo[] p = m.GetParameters();
                return m.Name == "PushHotkeyPressedBinding"
                    && p.Length >= 2
                    && p[0].ParameterType == typeof(string)
                    && p[1].ParameterType == typeof(Action)
                    && p.Skip(2).All(x => x.IsOptional);
            })
        ?? throw Missing("NHotkeyManager.PushHotkeyPressedBinding(string, Action)");

    /// <summary><c>manager.PushHotkeyPressedBinding(hotkey, action)</c> at top priority.</summary>
    public static void PushHotkeyPressedBinding(NHotkeyManager manager, string hotkey, Action action)
    {
        ParameterInfo[] p = PushHotkeyMethod.GetParameters();
        var args = new object?[p.Length];
        args[0] = hotkey;
        args[1] = action;
        for (int i = 2; i < p.Length; i++) args[i] = p[i].DefaultValue;
        Invoke(PushHotkeyMethod, manager, args);
    }

    // ── AutoSlay watchdog ───────────────────────────────────────────────

    /// <summary>
    /// A fresh watchdog with the game's default fuse. v0.113.0's constructor
    /// takes the log sinks (Watchdog.cs:62); they go to the mod log.
    /// </summary>
    public static Watchdog NewWatchdog()
    {
        ConstructorInfo ctor = typeof(Watchdog).GetConstructors()
            .Where(c => c.GetParameters().Skip(2).All(x => x.IsOptional))
            .OrderBy(c => c.GetParameters().Length)
            .FirstOrDefault()
            ?? throw Missing("Watchdog()");
        ParameterInfo[] p = ctor.GetParameters();
        var args = new object?[p.Length];
        for (int i = 0; i < p.Length; i++)
        {
            args[i] = p[i].ParameterType == typeof(Action<string>)
                ? new Action<string>(msg => Logger.Log(msg))
                : p[i].DefaultValue;
        }
        return (Watchdog)ctor.Invoke(args);
    }

    /// <summary>
    /// The static the game's own waits reset: <c>AutoSlayer.CurrentWatchdog</c>
    /// (v0.107.1 / v0.111.0) or <c>Watchdog.Current</c> (v0.113.0, private
    /// setter). Null when neither exists.
    /// </summary>
    public static readonly MethodInfo? SharedWatchdogSetter =
        typeof(AutoSlayer).GetProperty("CurrentWatchdog", BindingFlags.Public | BindingFlags.Static)?.GetSetMethod(true)
        ?? typeof(Watchdog).GetProperty("Current", BindingFlags.Public | BindingFlags.Static)?.GetSetMethod(true);

    private static object? Invoke(MethodInfo method, object? target, params object?[] args)
    {
        try
        {
            return method.Invoke(target, args);
        }
        catch (TargetInvocationException ex) when (ex.InnerException != null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }
    }

    private static IDisposable InvokeSelector(MethodInfo method, ICardSelector selector)
    {
        ParameterInfo[] p = method.GetParameters();
        var args = new object?[p.Length];
        args[0] = selector;
        // v0.111.0's localOnly defaults to false: the shared stack, which is
        // the only stack v0.107.1 has. In co-op the shared stack answers every
        // player's choice and skips PlayerChoiceSynchronizer; the local one
        // answers only ours and syncs it.
        for (int i = 1; i < p.Length; i++)
        {
            args[i] = p[i].Name == "localOnly" && SimMp.IsMultiplayer ? true : p[i].DefaultValue;
        }
        try
        {
            return (IDisposable)method.Invoke(null, args)!;
        }
        catch (TargetInvocationException ex) when (ex.InnerException != null)
        {
            // Callers catch the game's own exception type ("A card selector
            // is already active."), not the reflection wrapper.
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }
    }

    private static MethodInfo? SelectorMethod(string name) =>
        typeof(CardSelectCmd).GetMethods(Static)
            .FirstOrDefault(m =>
            {
                ParameterInfo[] p = m.GetParameters();
                return m.Name == name
                    && p.Length >= 1
                    && p[0].ParameterType == typeof(ICardSelector)
                    && p.Skip(1).All(x => x.IsOptional);
            });

    private static MissingMemberException Missing(string member) =>
        new($"Spire Mentor does not know this game build: {member} has an unexpected shape.");
}
