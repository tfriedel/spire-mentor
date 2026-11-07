// SimChoiceCapture.cs -- which option did the HUMAN pick on a non-combat
// decision screen? Reported as a v5 `human_action` (PROTOCOL.md section 5)
// so the advisor can grade it against the advice it computed for that screen.
//
// The advisor-mode screen poll (SimAdvisor.PollScreen) latches one decision
// at a time on (screen node, signature). When a decision latches we snapshot
// what the pick could change (deck, relics, potions, visited map nodes) and
// the option list; when it unlatches the record waits a few poll ticks for
// the change to land and then names the pick:
//
//   map_select                  visited map coordinate diff
//   card_reward, card_bundle    deck diff (count-based; nothing added = skip)
//   boss_relic, treasure        relic diff (nothing added = skip)
//   shop                        one purchase per signature change (card / relic
//                               / potion / removal diff); leaving is no choice
//   card_select (out of combat) the offered deck card that was removed or
//                               upgraded (by identity), else a card added
//   event                       Harmony prefix on NEventRoom.OptionButtonClicked
//   rest_site                   Harmony prefix on NRestSiteRoom.AfterSelectingOption
//
// The pure inference lives in ChoiceInference.cs (unit-tested). Anything the
// diff cannot pin to exactly one option reports NOTHING: a guessed index would
// grade the player on a move they never made. Picks the AI executed
// (SimAdvisor.ExecuteScreenChoice) are marked and never reported. In-combat
// selections (hand select, choose-a-card, combat card grids) are not captured
// in v5. Main thread only.

using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

internal static class SimChoiceCapture
{
    /// <summary>Poll ticks (0.25-0.5 s each) a decision's screen must be
    /// gone before a missing change is read as a skip / given up.</summary>
    private const int GoneTicksToResolve = 4;

    /// <summary>Hard cap on how long an unlatched record keeps looking for
    /// its change (a deck-select confirm, a slow map walk, a smith upgrade
    /// that finishes only after its card select).</summary>
    private const double MaxPendingSeconds = 120.0;

    private sealed class Record
    {
        public string Decision = "";
        public GodotObject? Screen;
        public string Signature = "";
        public RunState? Run;
        public List<string> Labels = new();
        public List<string> OptionIds = new();
        public List<(int col, int row)> MapCoords = new();
        public List<(string action, string id)> ShopOptions = new();
        public List<IReadOnlyList<string>> Bundles = new();
        public List<object?> Refs = new();
        public List<int> RefUpgradeLevels = new();
        public List<string> DeckBefore = new();
        public List<string> RelicsBefore = new();
        public List<string> PotionsBefore = new();
        public List<(int col, int row)> VisitedBefore = new();
        public string? AdviceRequestId;
        public long AdviceStateId = -1;
        public bool Peeked;
        public bool AiChose;
        public bool Done;
        public bool Unlatched;
        public int GoneTicks;
        public ulong UnlatchedAtMs;

        /// <summary>Event / rest site: reported by a Harmony hook, which may
        /// fire after the decision unlatched (a smith finishes only after its
        /// card select), so the record must stay findable until then.</summary>
        public bool ByHook =>
            Decision is NonCombatBridgeProtocol.EventState or NonCombatBridgeProtocol.RestSiteState;
    }

    private static Record? _current;
    private static readonly List<Record> Pending = new();

    // ------------------------------------------------------------------
    // Driven by SimAdvisor
    // ------------------------------------------------------------------

    /// <summary>
    /// Called by the screen poll on every tick with the decision it described
    /// and whether that is a NEW latch (screen or signature changed).
    /// </summary>
    public static void OnPoll(SimScreenDecision? now, bool changed)
    {
        try
        {
            if (changed)
            {
                if (_current != null)
                {
                    Unlatch(_current);
                    _current = null;
                }
                if (now != null)
                {
                    _current = Latch(now);
                    if (_current != null)
                    {
                        // A newer decision of the same kind is on screen: an
                        // older pending one must not read ITS changes. (For
                        // the shop this is the purchase that just changed the
                        // signature: already visible, so it resolves here.)
                        foreach (Record old in Pending)
                        {
                            if (!old.Done && old.Decision == _current.Decision && !old.ByHook)
                            {
                                Resolve(old, final: true);
                            }
                        }
                    }
                }
            }
            Tick();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimChoiceCapture] poll failed: {ex.Message}");
        }
    }

    /// <summary>The advice for (screen, signature) was accepted and rendered:
    /// the latched record for that decision is graded against it.</summary>
    public static void OnAdviceShown(GodotObject? screen, string signature, string requestId, long stateId)
    {
        Record? r = _current;
        if (r == null || !ReferenceEquals(r.Screen, screen) || r.Signature != signature)
        {
            return;
        }
        r.AdviceRequestId = requestId;
        r.AdviceStateId = stateId;
    }

    /// <summary>The F5 peek showed the AI's pick for the latched decision.</summary>
    public static void OnPeekShown()
    {
        if (_current != null)
        {
            _current.Peeked = true;
        }
    }

    /// <summary>The AI is about to click (screen, signature): whatever that
    /// decision resolves to is not a human pick.</summary>
    public static void MarkAiChoice(GodotObject? screen, string signature)
    {
        Record? r = _current;
        if (r != null && ReferenceEquals(r.Screen, screen) && r.Signature == signature)
        {
            r.AiChose = true;
        }
    }

    /// <summary>Forget everything (run restored, mode left Advisor): the
    /// screens torn down by a restore must not read as skips.</summary>
    public static void Reset()
    {
        _current = null;
        Pending.Clear();
    }

    // ------------------------------------------------------------------
    // Harmony entry points (main thread: both run off a button release)
    // ------------------------------------------------------------------

    /// <summary>NEventRoom.OptionButtonClicked(EventOption, int) prefix.</summary>
    public static void OnEventOptionClicked(EventOption option)
    {
        try
        {
            ReportByRef(NonCombatBridgeProtocol.EventState, option);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimChoiceCapture] event hook failed: {ex.Message}");
        }
    }

    /// <summary>NRestSiteRoom.AfterSelectingOption(RestSiteOption) prefix --
    /// runs only once the option succeeded (a cancelled smith never gets here).</summary>
    public static void OnRestOptionSelected(RestSiteOption option)
    {
        try
        {
            ReportByRef(NonCombatBridgeProtocol.RestSiteState, option);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimChoiceCapture] rest-site hook failed: {ex.Message}");
        }
    }

    // ------------------------------------------------------------------

    private static Record? Latch(SimScreenDecision d)
    {
        // potion_swap (v5.2): not graded yet -- an ungraded kind must not
        // fall through to the skip report when the loot screen closes.
        if (d.Local || d.AdviceFor is "game_over" or "crystal_sphere" or "reward_screen" or PotionSwapAdvice.AdviceFor)
        {
            return null;
        }
        // In-combat selections (hand select, choose-a-card, a combat card
        // grid) belong to the combat turn and are not graded in v5.
        if (SafeInCombat() || StrOf(d.State, "surface") == "choose_a_card")
        {
            return null;
        }
        RunState? run = RunManager.Instance.DebugOnlyGetState();
        Player? me = LocalContext.GetMe(run);
        if (run == null || me == null)
        {
            return null;
        }

        var r = new Record
        {
            Decision = d.AdviceFor,
            Screen = d.Screen,
            Signature = d.SignatureOrDefault,
            Run = run,
            DeckBefore = DeckIds(me),
            RelicsBefore = RelicIds(me),
            PotionsBefore = PotionIds(me),
            VisitedBefore = Visited(run),
        };

        switch (d.AdviceFor)
        {
            case "map_select":
                foreach (Dictionary<string, object> node in ListOf<Dictionary<string, object>>(d.State, "nodes"))
                {
                    int col = IntOf(node, "col");
                    int row = IntOf(node, "row");
                    r.MapCoords.Add((col, row));
                    r.Labels.Add($"{StrOf(node, "type")}@({col},{row})");
                }
                break;
            case NonCombatBridgeProtocol.CardRewardState:
                foreach (Dictionary<string, object?> card in ListOf<Dictionary<string, object?>>(d.State, "cards"))
                {
                    r.OptionIds.Add(StrOf(card, "id"));
                    r.Labels.Add(StrOf(card, "name"));
                }
                break;
            case NonCombatBridgeProtocol.CardBundleState:
                foreach (Dictionary<string, object> opt in ListOf<Dictionary<string, object>>(d.State, "options"))
                {
                    var ids = new List<string>();
                    var names = new List<string>();
                    if (opt.TryGetValue("cards", out object? cards) && cards is IEnumerable<Dictionary<string, object?>> list)
                    {
                        foreach (Dictionary<string, object?> c in list)
                        {
                            ids.Add(StrOf(c, "id"));
                            names.Add(StrOf(c, "name"));
                        }
                    }
                    r.Bundles.Add(ids);
                    r.Labels.Add(string.Join(" + ", names));
                }
                break;
            case NonCombatBridgeProtocol.BossRelicState:
            case NonCombatBridgeProtocol.TreasureState:
            case NonCombatBridgeProtocol.EventState:
            case NonCombatBridgeProtocol.RestSiteState:
                foreach (Dictionary<string, object> opt in OptionDicts(d.State))
                {
                    r.OptionIds.Add(StrOf(opt, "id"));
                    r.Labels.Add(StrOf(opt, "label"));
                }
                r.Refs = d.OptionRefs.ToList();
                break;
            case NonCombatBridgeProtocol.ShopState:
                foreach (Dictionary<string, object> opt in OptionDicts(d.State))
                {
                    string action = StrOf(opt, "action");
                    string label = StrOf(opt, "label");
                    r.ShopOptions.Add((action, label));
                    string name = opt.TryGetValue("item", out object? item) && item is Dictionary<string, object?> it
                        ? StrOf(it, "name")
                        : "";
                    r.Labels.Add(name.Length > 0 ? name : label);
                }
                break;
            case NonCombatBridgeProtocol.CardSelectState:
                // One pick per report: a multi-card select has no single index.
                if (IntOf(d.State, "max_select") > 1)
                {
                    return null;
                }
                r.Labels = d.OptionLabels.ToList();
                r.Refs = d.OptionRefs.ToList();
                r.RefUpgradeLevels = r.Refs.Select(o => o is CardModel c ? SafeUpgradeLevel(c) : -1).ToList();
                foreach (Dictionary<string, object?> card in ListOf<Dictionary<string, object?>>(d.State, "cards"))
                {
                    r.OptionIds.Add(StrOf(card, "id"));
                }
                break;
            default:
                return null;
        }
        return r;
    }

    private static void Unlatch(Record r)
    {
        r.Unlatched = true;
        r.UnlatchedAtMs = Time.GetTicksMsec();
        if (r.AiChose || r.Done)
        {
            return;
        }
        Pending.Add(r);
    }

    private static void Tick()
    {
        if (Pending.Count == 0)
        {
            return;
        }
        foreach (Record r in Pending.ToList())
        {
            if (r.Done || r.AiChose)
            {
                Pending.Remove(r);
                continue;
            }
            if (!ReferenceEquals(RunManager.Instance.DebugOnlyGetState(), r.Run))
            {
                Pending.Remove(r); // run ended or was restored
                continue;
            }
            if (ScreenGone(r.Screen))
            {
                r.GoneTicks++;
            }
            bool timedOut = Time.GetTicksMsec() - r.UnlatchedAtMs > (ulong)(MaxPendingSeconds * 1000);
            Resolve(r, final: r.GoneTicks >= GoneTicksToResolve || timedOut);
            if (r.Done || timedOut)
            {
                Pending.Remove(r);
            }
        }
    }

    /// <summary>Try to name the pick. A positive pick reports at once; a skip
    /// only on the <paramref name="final"/> pass (the change may still be
    /// landing); <see cref="ChoiceInference.Unknown"/> never reports.</summary>
    private static void Resolve(Record r, bool final)
    {
        if (r.Done)
        {
            return;
        }
        if (r.ByHook)
        {
            // Reported by their hooks; the pending entry only keeps the
            // option refs findable until the hook fires (or the record times
            // out in Tick). The room can be hidden under a smith's card
            // select for longer than the gone-window, so "gone" ends nothing.
            return;
        }
        RunState? run = RunManager.Instance.DebugOnlyGetState();
        Player? me = LocalContext.GetMe(run);
        if (run == null || me == null)
        {
            return;
        }

        int? pick = r.Decision switch
        {
            "map_select" => ChoiceInference.MapPick(
                r.MapCoords, ChoiceInference.NewlyVisited(r.VisitedBefore, Visited(run))),
            NonCombatBridgeProtocol.CardRewardState =>
                ChoiceInference.AddedCardPick(r.OptionIds, r.DeckBefore, DeckIds(me)),
            NonCombatBridgeProtocol.CardBundleState =>
                ChoiceInference.BundlePick(r.Bundles, r.DeckBefore, DeckIds(me)),
            NonCombatBridgeProtocol.BossRelicState or NonCombatBridgeProtocol.TreasureState =>
                ChoiceInference.AddedRelicPick(r.OptionIds, r.RelicsBefore, RelicIds(me)),
            NonCombatBridgeProtocol.ShopState => ChoiceInference.ShopPick(
                r.ShopOptions, r.DeckBefore, DeckIds(me), r.RelicsBefore, RelicIds(me),
                r.PotionsBefore, PotionIds(me)),
            NonCombatBridgeProtocol.CardSelectState => DeckSelectPick(r, me),
            _ => ChoiceInference.Unknown,
        };

        if (pick is int i && i >= 0)
        {
            Report(r, i);
            return;
        }
        // A skip needs the screen actually CLOSED, not just gone from view: the
        // overlay stack also hides a live screen (AfterOverlayHidden) when
        // something is pushed over it, and in co-op that happened when a
        // teammate picked their reward -- the unchanged deck then graded as
        // "you skipped" while the player was still choosing.
        if (pick == null && final && ScreenClosed(r.Screen))
        {
            Report(r, null);
            return;
        }
        if (final)
        {
            r.Done = true; // nothing definite: say nothing
        }
    }

    /// <summary>Out-of-combat deck select: the one offered card that left the
    /// deck or gained an upgrade (by identity), else one offered card id that
    /// joined. Never a skip: an enchant or a cancel leaves no trace we can
    /// read, and "no change" must not be graded as a decision.</summary>
    private static int? DeckSelectPick(Record r, Player me)
    {
        // By identity: a card model's equality is not ours to rely on.
        var deck = new HashSet<object>(me.Deck.Cards, ReferenceEqualityComparer.Instance);
        var changed = new List<bool>();
        for (int i = 0; i < r.Refs.Count; i++)
        {
            bool c = r.Refs[i] is CardModel card &&
                     (!deck.Contains(card) || SafeUpgradeLevel(card) > r.RefUpgradeLevels[i]);
            changed.Add(c);
        }
        int? byRef = ChoiceInference.SingleChanged(changed);
        if (byRef is int k && k >= 0)
        {
            return k;
        }
        int? added = ChoiceInference.AddedCardPick(r.OptionIds, r.DeckBefore, DeckIds(me));
        return added is int a && a >= 0 ? a : ChoiceInference.Unknown;
    }

    private static void ReportByRef(string decision, object option)
    {
        Record? r = _current != null && _current.Decision == decision && RefIndex(_current, option) >= 0
            ? _current
            : Pending.LastOrDefault(p => !p.Done && p.Decision == decision && RefIndex(p, option) >= 0);
        if (r == null || r.Done || r.AiChose)
        {
            return;
        }
        int index = RefIndex(r, option);
        if (index < 0)
        {
            return;
        }
        Report(r, index);
    }

    private static int RefIndex(Record r, object option)
    {
        for (int i = 0; i < r.Refs.Count; i++)
        {
            if (ReferenceEquals(r.Refs[i], option))
            {
                return i;
            }
        }
        return -1;
    }

    private static void Report(Record r, int? index)
    {
        r.Done = true;
        if (r.AiChose)
        {
            return;
        }
        string label = index is int i
            ? (i < r.Labels.Count && r.Labels[i].Length > 0 ? r.Labels[i] : $"option {i}")
            : "Skip";
        SimAdvisor.Instance.SendChoice(r.Decision, index, label, r.AdviceRequestId, r.AdviceStateId, r.Peeked);
    }

    // ------------------------------------------------------------------
    // Game reads (never throw)
    // ------------------------------------------------------------------

    private static bool SafeInCombat()
    {
        try
        {
            return CombatManager.Instance?.IsInProgress ?? false;
        }
        catch
        {
            return false;
        }
    }

    private static bool ScreenGone(GodotObject? screen)
    {
        if (screen == null || !GodotObject.IsInstanceValid(screen))
        {
            return true;
        }
        if (screen is Node node && !node.IsInsideTree())
        {
            return true;
        }
        return screen is CanvasItem ci && !ci.IsVisibleInTree();
    }

    /// <summary>Freed or leaving the tree -- a real close. Unlike
    /// <see cref="ScreenGone"/>, a hidden screen is not closed.</summary>
    private static bool ScreenClosed(GodotObject? screen)
    {
        if (screen == null || !GodotObject.IsInstanceValid(screen))
        {
            return true;
        }
        return screen is Node node && (!node.IsInsideTree() || node.IsQueuedForDeletion());
    }

    private static List<string> DeckIds(Player me)
    {
        try
        {
            return me.Deck.Cards.Select(c => c.Id.Entry).ToList();
        }
        catch
        {
            return new List<string>();
        }
    }

    private static List<string> RelicIds(Player me)
    {
        try
        {
            return me.Relics.Select(r => r.Id.Entry).ToList();
        }
        catch
        {
            return new List<string>();
        }
    }

    private static List<string> PotionIds(Player me)
    {
        try
        {
            return me.PotionSlots.Where(p => p != null).Select(p => p!.Id.Entry).ToList();
        }
        catch
        {
            return new List<string>();
        }
    }

    private static List<(int col, int row)> Visited(RunState run)
    {
        try
        {
            return run.VisitedMapCoords.Select(c => (c.col, c.row)).ToList();
        }
        catch
        {
            return new List<(int col, int row)>();
        }
    }

    private static int SafeUpgradeLevel(CardModel card)
    {
        try
        {
            return card.CurrentUpgradeLevel;
        }
        catch
        {
            return -1;
        }
    }

    private static IEnumerable<T> ListOf<T>(IReadOnlyDictionary<string, object> state, string key) =>
        state.TryGetValue(key, out object? v) && v is IEnumerable<T> list ? list : Enumerable.Empty<T>();

    /// <summary>`options` is a List of Dictionary&lt;string, object&gt; on most
    /// screens and of Dictionary&lt;string, object?&gt; on events.</summary>
    private static IEnumerable<Dictionary<string, object>> OptionDicts(IReadOnlyDictionary<string, object> state)
    {
        if (!state.TryGetValue("options", out object? v) || v is not System.Collections.IEnumerable list)
        {
            yield break;
        }
        foreach (object? o in list)
        {
            if (o is Dictionary<string, object> d)
            {
                yield return d;
            }
            else if (o is Dictionary<string, object?> dn)
            {
                yield return dn.ToDictionary(kv => kv.Key, kv => kv.Value ?? "");
            }
        }
    }

    private static string StrOf<TV>(IReadOnlyDictionary<string, TV> d, string key) =>
        d.TryGetValue(key, out TV? v) && v != null ? v.ToString() ?? "" : "";

    private static int IntOf<TV>(IReadOnlyDictionary<string, TV> d, string key) =>
        d.TryGetValue(key, out TV? v) && v is int i ? i : -1;
}
