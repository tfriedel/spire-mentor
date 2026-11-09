// SimAiPartnerChoices.cs -- an AI partner's run-level picks and prompt answers
// (docs/plans/2026-11-05-ai-coop-partners.md, phase 2).
//
// A partner has no screens: the host never renders them. So where the human's
// advisor lane scrapes a payload off a live screen (SimScreenAdvice), the
// partner builds the same payload from model state -- its own EventModel, its
// rest site options, the cards of its CardReward -- and asks its own advisor
// (SimAiPartnerAdvisor), then answers the way a remote client would.
//
// What the advisor answers: card rewards (also a fight's choose-a-card
// screen), events, rest sites, treasure and relic picks. What a heuristic
// answers, for now: deck prompts (upgrade / remove / transform: the
// advisor's card_select surface keys on a prompt_key the partner does not
// capture yet) and in-combat hand prompts. Every ask falls back to the
// heuristic on no advisor, a timeout or an unusable reply, so a partner
// never stalls the run.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

/// <summary>The prompt a remote choice answers: the method that opened it
/// and its arguments (SimAiPartnerPatches.PromptPrefix).</summary>
internal sealed record SimAiPartnerPrompt(string Kind, object?[] Args);

internal enum SimAskState
{
    Pending,
    Answered,
    Unavailable,
}

internal sealed class SimAiPartnerChoices
{
    /// <summary>How long a screen ask may go unanswered (the run policy
    /// answers in well under a second; this covers a busy advisor).</summary>
    private static readonly TimeSpan ScreenTimeout = TimeSpan.FromSeconds(20);

    private readonly SimAiPartner _partner;
    private readonly SimAiPartnerBrain _brain;
    private readonly object _lock = new();
    private string? _inflightKey;
    private DateTime _inflightAt;
    private (string Key, SimAdvice Advice)? _reply;
    private readonly HashSet<string> _gaveUp = new();

    /// <summary>The latest prompt opened for this partner, not yet bound to
    /// a choice id.</summary>
    public SimAiPartnerPrompt? Prompt;

    /// <summary>The host's remote-choice waits for this partner, each with
    /// the prompt that opened it. A prompt opens (the From* / OnSelect
    /// prefix) and then waits for its choice id synchronously, so the wait
    /// takes the prompt opened just before it; a later prompt cannot
    /// relabel an earlier wait.</summary>
    private readonly Queue<(uint Id, SimAiPartnerPrompt? Prompt)> _pendingChoices = new();

    public SimAiPartnerChoices(SimAiPartner partner, SimAiPartnerBrain brain)
    {
        _partner = partner;
        _brain = brain;
    }

    public void OnRemoteChoiceAwaited(uint choiceId)
    {
        _pendingChoices.Enqueue((choiceId, Prompt));
        Prompt = null;
    }

    public bool HasPendingChoice => _pendingChoices.Count > 0;

    // ---- the generic ask -------------------------------------------------------

    /// <summary>
    /// The advisor's pick for one decision (<paramref name="key"/>):
    /// <see cref="SimAskState.Answered"/> with the chosen option's index (null
    /// = skip), <see cref="SimAskState.Pending"/> while it thinks,
    /// <see cref="SimAskState.Unavailable"/> to fall back.
    /// </summary>
    public (SimAskState State, int? Index) Ask(Player me, string key, string adviceFor, Func<Dictionary<string, object>?> build)
    {
        SimAiPartnerAdvisor? advisor = _partner.Advisor;
        if (advisor == null || !advisor.Connected || _gaveUp.Contains(key))
        {
            return (SimAskState.Unavailable, null);
        }
        lock (_lock)
        {
            if (_reply is { } r && r.Key == key)
            {
                _reply = null;
                _inflightKey = null;
                SimBestAction? best = r.Advice.Best;
                if (best?.Action == "skip")
                {
                    return (SimAskState.Answered, null);
                }
                if (best?.Action == "choose" && best.Index >= 0)
                {
                    return (SimAskState.Answered, best.Index);
                }
                _gaveUp.Add(key);
                return (SimAskState.Unavailable, null);
            }
        }
        if (_inflightKey == key)
        {
            if (DateTime.UtcNow - _inflightAt < ScreenTimeout)
            {
                return (SimAskState.Pending, null);
            }
            Logger.Log($"[SimAiPartner] {_partner.Label}: no {adviceFor} advice within {ScreenTimeout.TotalSeconds:0} s; deciding without it.");
            _gaveUp.Add(key);
            _inflightKey = null;
            return (SimAskState.Unavailable, null);
        }
        Dictionary<string, object>? state = _brain.BuildAsMe(me, build);
        if (state == null)
        {
            _gaveUp.Add(key);
            return (SimAskState.Unavailable, null);
        }
        state["type"] = "advice_request";
        state["advice_for"] = adviceFor;
        // Same brain as the player's advice (SimAiPartnerBrain's combat asks).
        state["brain"] = SimAdvisor.CurrentBrain;
        _inflightKey = key;
        _inflightAt = DateTime.UtcNow;
        if (!advisor.Request(state, root => OnReply(key, root)))
        {
            _inflightKey = null;
            return (SimAskState.Unavailable, null);
        }
        return (SimAskState.Pending, null);
    }

    private void OnReply(string key, JsonElement root)
    {
        try
        {
            if (root.TryGetProperty("advice", out JsonElement adviceEl))
            {
                SimAdvice advice = SimAdvice.Parse(adviceEl);
                lock (_lock)
                {
                    _reply = (key, advice);
                }
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartner] {_partner.Label}: unreadable advice: {ex.Message}");
        }
    }

    // ---- payloads, built from model state ------------------------------------------

    /// <summary>The card_reward surface (SimScreenAdvice.DescribeCardReward's
    /// shape) for cards offered to this partner.</summary>
    internal static Dictionary<string, object> CardRewardState(Player me, IReadOnlyList<CardModel> cards, bool canSkip, string? surface)
    {
        var rows = new List<Dictionary<string, object?>>();
        for (int i = 0; i < cards.Count; i++)
        {
            Dictionary<string, object?> data = SimStateExtractor.SerializeCard(cards[i]);
            data["index"] = i;
            rows.Add(data);
        }
        var state = new Dictionary<string, object>
        {
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.CardRewardState,
            ["cards"] = rows,
            ["can_skip"] = canSkip,
        };
        if (surface != null)
        {
            state["surface"] = surface;
        }
        SimStateExtractor.ApplyEnvelope(state, me.RunState as RunState, me);
        return state;
    }

    /// <summary>A choose-a-card screen as the host's own (SimScreenAdvice
    /// DescribeChooseACard): in a fight it carries the partner's board and
    /// what opened it (Discovery, a potion), so the partner's advisor answers
    /// it on its fight mirror like the host's; out of a fight (Lead
    /// Paperweight) it is a deck pick and <c>combat</c> is null.</summary>
    internal static Dictionary<string, object> ChooseACardState(Player me, IReadOnlyList<CardModel> cards, bool canSkip, object? source)
    {
        Dictionary<string, object> state = CardRewardState(me, cards, canSkip, "choose_a_card");
        state["combat"] = (object?)SimCombatHandler.BuildEmbeddedCombatState(me)!;
        string? sourceModel = SimCardSelectContext.EntryOf(source);
        if (sourceModel != null)
        {
            state["source"] = "FromChooseACardScreen";
            state["prompt_key"] = "TO_OBTAIN";
            state["source_model"] = sourceModel;
            state["source_kind"] = (object?)SimCardSelectContext.KindOfModel(source)!;
        }
        return state;
    }

    /// <summary>A one-card deck screen (upgrade / transform / remove) as the
    /// host's card_select (SimCardSelector): the prompt key says what it
    /// does, which is what both brains read.</summary>
    internal static Dictionary<string, object> DeckSelectState(Player me, IReadOnlyList<CardModel> cards, CardSelectorPrefs? prefs, string kind)
    {
        var rows = new List<Dictionary<string, object?>>();
        for (int i = 0; i < cards.Count; i++)
        {
            Dictionary<string, object?> data = SimStateExtractor.SerializeCard(cards[i]);
            data["index"] = i;
            rows.Add(data);
        }
        string method = kind.Contains('.') ? kind[(kind.LastIndexOf('.') + 1)..] : kind;
        var state = new Dictionary<string, object>
        {
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.CardSelectState,
            ["cards"] = rows,
            ["min_select"] = 1,
            ["max_select"] = 1,
            ["min_count"] = 1,
            ["max_count"] = 1,
            ["cancelable"] = prefs?.Cancelable ?? false,
            ["prompt_key"] = (object?)prefs?.Prompt.LocEntryKey!,
            ["source"] = method,
            ["combat"] = (object?)SimCombatHandler.BuildEmbeddedCombatState(me)!,
        };
        SimStateExtractor.ApplyEnvelope(state, me.RunState as RunState, me);
        return state;
    }

    /// <summary>The boss_relic surface (also the chest's) for relics offered
    /// to this partner.</summary>
    internal static Dictionary<string, object> RelicState(Player me, IReadOnlyList<RelicModel> relics, string adviceFor, string? surface)
    {
        var state = new Dictionary<string, object>
        {
            [NonCombatBridgeProtocol.TypeField] = adviceFor,
            ["options"] = relics.Select((r, i) => SimRelicOptions.Option(i, NonCombatBridgeProtocol.CollectTreasureAction, r)).ToList(),
        };
        if (surface != null)
        {
            state["surface"] = surface;
        }
        SimStateExtractor.ApplyEnvelope(state, me.RunState as RunState, me);
        return state;
    }

    // ---- prompt answers ----------------------------------------------------------

    /// <summary>
    /// Answers the host's remote-choice waits for this partner, one per tick
    /// (PlayerChoiceMessage; the result type must match what the prompt's
    /// caller reads back, or the host throws).
    /// </summary>
    public void Tick(Player me)
    {
        if (_pendingChoices.Count == 0)
        {
            return;
        }
        (uint id, SimAiPartnerPrompt? prompt) = _pendingChoices.Peek();
        string kind = prompt?.Kind ?? "?";
        PlayerChoiceResult? result = Answer(me, id, prompt, out string how);
        if (result == null)
        {
            // Waiting for the advisor.
            return;
        }
        _pendingChoices.Dequeue();
        Logger.Log($"[SimAiPartner] {_partner.Label} answers choice {id} ({kind}): {how}.");
        _partner.Send(new PlayerChoiceMessage { choiceId = id, result = result.ToNetData() });
    }

    private PlayerChoiceResult? Answer(Player me, uint id, SimAiPartnerPrompt? prompt, out string how)
    {
        how = "none";
        string kind = prompt?.Kind ?? "";
        object?[] args = prompt?.Args ?? Array.Empty<object?>();
        CardSelectorPrefs? prefs = args.OfType<CardSelectorPrefs>().Cast<CardSelectorPrefs?>().FirstOrDefault();
        int minSelect = prefs?.MinSelect ?? 0;

        if (kind == "CardReward.OnSelect" && args.FirstOrDefault() is CardReward reward)
        {
            List<CardModel> cards = reward.Cards.ToList();
            return AskIndex(me, $"cardreward:{id}", NonCombatBridgeProtocol.CardRewardState,
                () => CardRewardState(me, cards, reward.CanSkip, null), cards.Count, reward.CanSkip ? null : 0, cards, out how);
        }
        if (kind.EndsWith(".FromChooseACardScreen", StringComparison.Ordinal) &&
            args.OfType<IReadOnlyList<CardModel>>().FirstOrDefault() is { } offered)
        {
            bool canSkip = args.OfType<bool>().LastOrDefault();
            List<CardModel> cards = offered.ToList();
            object? source = SimCardSelectContext.SourceObjectFrom(args);
            return AskIndex(me, $"chooseacard:{id}", NonCombatBridgeProtocol.CardRewardState,
                () => ChooseACardState(me, cards, canSkip, source), cards.Count, canSkip ? null : 0, cards, out how);
        }
        if (kind.EndsWith(".FromChooseARelicScreen", StringComparison.Ordinal) &&
            args.OfType<IReadOnlyList<RelicModel>>().FirstOrDefault() is { } relics)
        {
            List<RelicModel> list = relics.ToList();
            (SimAskState state, int? index) = Ask(me, $"relic:{id}", NonCombatBridgeProtocol.BossRelicState,
                () => RelicState(me, list, NonCombatBridgeProtocol.BossRelicState, null));
            if (state == SimAskState.Pending)
            {
                return null;
            }
            int pick = state == SimAskState.Answered && index is int i && i < list.Count ? i : 0;
            how = $"{list.ElementAtOrDefault(pick)?.Id.Entry} ({(state == SimAskState.Answered ? "advisor" : "first")})";
            return PlayerChoiceResult.FromIndex(list.Count > 0 ? pick : null);
        }
        if (kind.EndsWith(".FromChooseABundleScreen", StringComparison.Ordinal))
        {
            how = "bundle 0 (first)";
            return PlayerChoiceResult.FromIndex(0);
        }
        if (kind.Contains("SimpleGrid", StringComparison.Ordinal))
        {
            List<CardModel> grid = args.OfType<IEnumerable<CardModel>>().FirstOrDefault()?.ToList() ?? new List<CardModel>();
            List<int> picks = Rank(grid, kind).Take(minSelect).Select(c => grid.IndexOf(c)).ToList();
            how = picks.Count == 0 ? "none" : string.Join(",", picks.Select(i => grid[i].Id.Entry));
            return PlayerChoiceResult.FromIndexes(picks);
        }
        if (kind.Contains("FromDeck", StringComparison.Ordinal))
        {
            List<CardModel> deck = me.Deck.Cards.ToList();
            IEnumerable<CardModel> candidates = kind.EndsWith("FromDeckForUpgrade", StringComparison.Ordinal)
                ? deck.Where(c => c.IsUpgradable)
                : deck;
            // An upgrade is always worth taking (a smith, an event's upgrade);
            // other deck prompts take what they must.
            int count = kind.EndsWith("FromDeckForUpgrade", StringComparison.Ordinal) ? Math.Max(1, minSelect) : minSelect;
            List<CardModel> pool = candidates.ToList();
            if (count == 1 && pool.Count > 1)
            {
                // One card to upgrade / transform / remove: the advisor's
                // deck screen (the Smith, Neow's New Leaf), as the host's own
                // card_select. The partner played these by the hand-written
                // order below until 0.10.0-beta.7.
                (SimAskState asked, int? index) = Ask(me, $"deck:{id}", NonCombatBridgeProtocol.CardSelectState,
                    () => DeckSelectState(me, pool, prefs, kind));
                if (asked == SimAskState.Pending)
                {
                    return null;
                }
                if (asked == SimAskState.Answered && index is int i && i >= 0 && i < pool.Count)
                {
                    how = $"{pool[i].Id.Entry} (advisor)";
                    return PlayerChoiceResult.FromMutableDeckCards(new List<CardModel> { pool[i] });
                }
            }
            List<CardModel> picks = Rank(pool, kind).Take(count).ToList();
            how = picks.Count == 0 ? "none" : string.Join(",", picks.Select(c => c.Id.Entry)) + " (heuristic)";
            return PlayerChoiceResult.FromMutableDeckCards(picks);
        }
        if (kind.Contains("FromHand", StringComparison.Ordinal) || kind.Contains("FromCombatPile", StringComparison.Ordinal))
        {
            List<CardModel> pool = args.OfType<IEnumerable<CardModel>>().FirstOrDefault()?.ToList()
                                   ?? PileType.Hand.GetPile(me).Cards.ToList();
            List<CardModel> picks = Rank(pool, kind).Take(minSelect).ToList();
            how = picks.Count == 0 ? "none" : string.Join(",", picks.Select(c => c.Id.Entry)) + " (heuristic)";
            return PlayerChoiceResult.FromMutableCombatCards(picks);
        }
        how = "none (unknown prompt)";
        return PlayerChoiceResult.FromIndex(null);
    }

    /// <summary>A card pick by the advisor, else <paramref name="fallback"/>
    /// (null = skip). Null while the advisor thinks.</summary>
    private PlayerChoiceResult? AskIndex(Player me, string key, string adviceFor, Func<Dictionary<string, object>?> build,
        int count, int? fallback, List<CardModel> cards, out string how)
    {
        (SimAskState state, int? index) = Ask(me, key, adviceFor, build);
        if (state == SimAskState.Pending)
        {
            how = "";
            return null;
        }
        int? pick = state == SimAskState.Answered ? index : fallback;
        if (pick is int i && (i < 0 || i >= count))
        {
            pick = fallback;
        }
        how = (pick is int p ? cards[p].Id.Entry : "skip") + (state == SimAskState.Answered ? " (advisor)" : " (fallback)");
        return PlayerChoiceResult.FromIndex(pick);
    }

    /// <summary>The heuristic order for a forced pick: for an upgrade, the
    /// cards that are not starter Strikes/Defends first; for anything else
    /// (remove, transform, discard, exhaust), the junk first -- curses,
    /// statuses, then the starter Strikes and Defends.</summary>
    private static IEnumerable<CardModel> Rank(IEnumerable<CardModel> cards, string kind)
    {
        bool upgrade = kind.Contains("Upgrade", StringComparison.Ordinal);
        return cards.OrderBy(c => upgrade ? (IsStarterBasic(c) ? 1 : 0) : JunkRank(c));
    }

    private static int JunkRank(CardModel c) =>
        c.Type == CardType.Curse ? 0 :
        c.Type == CardType.Status ? 1 :
        IsStarterBasic(c) && c.Id.Entry.StartsWith("STRIKE", StringComparison.Ordinal) ? 2 :
        IsStarterBasic(c) ? 3 : 4;

    private static bool IsStarterBasic(CardModel c) =>
        c.Id.Entry.StartsWith("STRIKE", StringComparison.Ordinal) || c.Id.Entry.StartsWith("DEFEND", StringComparison.Ordinal);
}
