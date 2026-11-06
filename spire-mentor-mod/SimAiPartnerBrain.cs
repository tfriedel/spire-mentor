// SimAiPartnerBrain.cs -- what an AI partner decides, and when (SPIKE).
//
// Polled on the main thread (SimAiPartner, every 0.25 s). Each step reads
// the HOST's state for the partner's Player and, when the host is waiting on
// that player, sends the one message a remote client would send. The spike
// plays the dumbest valid game: the first playable card at the first
// hittable enemy, then end turn; the first option or a skip elsewhere.
//
// Every decision is sent at most once per "key" (turn, choice id, vote
// round...), since the host's state only changes after the message has
// round-tripped through the host's message loop.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using HarmonyLib;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game.Sync;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

internal sealed class SimAiPartnerBrain
{
    /// <summary>How long a sent play may stay unresolved before the partner
    /// looks again (a refused play leaves the hand unchanged).</summary>
    private static readonly TimeSpan PlayTimeout = TimeSpan.FromSeconds(3);

    /// <summary>Plays per turn before the partner just ends it (a loop guard:
    /// a card the host keeps refusing).</summary>
    private const int MaxPlaysPerTurn = 15;

    private readonly SimAiPartner _partner;

    private string? _pendingPlayKey;
    private DateTime _pendingSince;
    private int _turnSeen = -1;
    private int _playsThisTurn;
    private int _endTurnSentFor = -1;
    private DateTime _endTurnSentAt;
    private static readonly TimeSpan EndTurnRetry = TimeSpan.FromSeconds(5);
    private readonly HashSet<string> _sent = new();

    public SimAiPartnerBrain(SimAiPartner partner)
    {
        _partner = partner;
    }

    public Player? Me
    {
        get
        {
            RunState? state = RunManager.Instance.DebugOnlyGetState();
            return state?.GetPlayer(_partner.NetId);
        }
    }

    public void Tick()
    {
        if (!RunManager.Instance.IsInProgress)
        {
            return;
        }
        Player? me = Me;
        // A loaded run is in progress a tick before its location buffer is
        // set up (continue-1: every step reads Location).
        if (me == null || RunManager.Instance.RunLocationTargetedBuffer == null)
        {
            return;
        }
        // Choice prompts first: the host blocks inside the prompt (an event
        // option's card pick, a hand select mid-card) until it is answered.
        Step("choice", () => TickChoice(me));
        Step("rewards", () => TickRewards(me));
        if (CombatManager.Instance.IsInProgress)
        {
            Step("combat", () => TickCombat(me));
            return;
        }
        Waiting = PartnerWait.None;
        // One screen at a time, as a player has: no room decision while one
        // of our prompts or reward sets is still open. Two of our prompts open
        // at once (a card reward's pick, then a smith's upgrade) once had the
        // upgrade's deck card answer the card reward: the card reward threw
        // on every peer and the run diverged (mixed-1, plan phase 4).
        if (Choices.HasPendingChoice || _openRewardSets.Count > 0)
        {
            return;
        }
        Step("event", () => TickEvent(me));
        Step("rest", () => TickRestSite(me));
        Step("treasure", () => TickTreasure(me));
        Step("act", TickActChange);
        Step("map", () => TickMapVote(me));
    }

    private readonly HashSet<string> _loggedFailures = new();

    /// <summary>One step's failure never blocks the others; each distinct
    /// failure is logged once.</summary>
    private void Step(string name, Action step)
    {
        try
        {
            step();
        }
        catch (Exception ex)
        {
            string key = $"{name}:{ex.GetType().Name}:{ex.Message}";
            if (_loggedFailures.Add(key))
            {
                Logger.Log($"[SimAiPartner] {_partner.Label} {name} step failed: {ex}");
            }
        }
    }

    // ---- choice prompts --------------------------------------------------------

    /// <summary>Prompt answers and screen asks (SimAiPartnerChoices).</summary>
    public SimAiPartnerChoices Choices => _choices ??= new SimAiPartnerChoices(_partner, this);
    private SimAiPartnerChoices? _choices;

    public void OnRemoteChoiceAwaited(uint choiceId) => Choices.OnRemoteChoiceAwaited(choiceId);

    private void TickChoice(Player me) => Choices.Tick(me);

    // ---- rewards ---------------------------------------------------------------

    /// <summary>One of our reward sets being worked through: the next reward
    /// to look at, and when the last selection went out.</summary>
    private sealed class OpenRewardSet
    {
        public required RewardsSet Set;
        public int Next;
        /// <summary>Where the set was offered: its messages target that room
        /// even after the host has moved on (the host dispatches a message for
        /// a location it has visited at once).</summary>
        public RunLocation OfferedAt;
        public DateTime LastSentAt;
    }

    private readonly Queue<OpenRewardSet> _openRewardSets = new();

    /// <summary>Gap between two selections from one set: the host runs each
    /// reward's OnSelect (a card reward waits there for our pick).</summary>
    private static readonly TimeSpan RewardGap = TimeSpan.FromMilliseconds(600);

    public void OnRewardsSetOffered(RewardsSet set) =>
        _openRewardSets.Enqueue(new OpenRewardSet { Set = set, OfferedAt = Location });

    /// <summary>
    /// The host offers our reward sets without waiting on them (CombatRoom);
    /// a real client takes what it wants (RewardSelectedMessage per reward:
    /// gold, a relic, a potion it has room for, the card reward -- whose
    /// card the advisor then picks at the prompt) and closes the rest
    /// (RewardSetSkippedMessage). One selection per tick, none while one of
    /// our prompts is open.
    /// </summary>
    private void TickRewards(Player me)
    {
        if (_openRewardSets.Count == 0 || Choices.HasPendingChoice)
        {
            return;
        }
        OpenRewardSet open = _openRewardSets.Peek();
        if (DateTime.UtcNow - open.LastSentAt < RewardGap)
        {
            return;
        }
        List<Reward> rewards = open.Set.Rewards;
        while (open.Next < rewards.Count)
        {
            int index = open.Next++;
            Reward reward = rewards[index];
            if (!Wanted(me, reward))
            {
                continue;
            }
            Logger.Log($"[SimAiPartner] {_partner.Label} takes reward {index} ({reward.GetType().Name}) of set {open.Set.Id}.");
            _partner.Send(new RewardSelectedMessage { setId = open.Set.Id, rewardIndex = index, location = open.OfferedAt });
            open.LastSentAt = DateTime.UtcNow;
            return;
        }
        _openRewardSets.Dequeue();
        if (open.Set.AllRewardsSuccessfullySelected)
        {
            // Everything taken: the set completes by itself, and a skip
            // would make the host throw.
            return;
        }
        Logger.Log($"[SimAiPartner] {_partner.Label} closes reward set {open.Set.Id}.");
        _partner.Send(new RewardSetSkippedMessage { setId = open.Set.Id, location = open.OfferedAt });
    }

    private static bool Wanted(Player me, Reward reward) => reward switch
    {
        GoldReward => true,
        RelicReward => true,
        CardReward => true,
        PotionReward potion => SimPotionRules.CanProcure(me, potion.Potion),
        // Card removal (a deck prompt) and special cards: not yet.
        _ => false,
    };

    // ---- rest site, treasure, act change, map ---------------------------------

    /// <summary>
    /// v0.111.0's RestSiteRoom.Exit waits for every player's rest site. The
    /// advisor picks among the options the human's lane offers (enabled, no
    /// Mend: it heals a teammate through a picker); heal without it. A smith
    /// opens a deck prompt for us, answered in SimAiPartnerChoices.
    /// </summary>
    private void TickRestSite(Player me)
    {
        if (RunManager.Instance.DebugOnlyGetState()?.CurrentRoom is not RestSiteRoom)
        {
            return;
        }
        string key = $"rest:{Location}";
        if (_sent.Contains(key))
        {
            return;
        }
        IReadOnlyList<RestSiteOption> all = RunManager.Instance.RestSiteSynchronizer.GetOptionsForPlayer(me);
        var offered = Enumerable.Range(0, all.Count)
            .Where(i => all[i].IsEnabled && all[i].OptionId != "MEND")
            .ToList();
        if (offered.Count == 0)
        {
            return;
        }
        (SimAskState state, int? index) = Choices.Ask(me, key, NonCombatBridgeProtocol.RestSiteState, () =>
        {
            var s = new Dictionary<string, object>
            {
                [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.RestSiteState,
                ["options"] = offered.Select(i => (object)new Dictionary<string, object>
                {
                    ["index"] = i,
                    ["id"] = all[i].OptionId,
                    ["action"] = NonCombatBridgeProtocol.RestOptionAction,
                    ["label"] = all[i].Title.GetFormattedText(),
                    ["description"] = all[i].Description.GetFormattedText(),
                    ["enabled"] = true,
                }).ToList(),
            };
            SimStateExtractor.ApplyEnvelope(s, me.RunState as RunState, me);
            return s;
        });
        if (state == SimAskState.Pending)
        {
            return;
        }
        int heal = offered.FirstOrDefault(i => all[i] is HealRestSiteOption, offered[0]);
        int pick = state == SimAskState.Answered && index is int i && offered.Contains(i) ? i : heal;
        Once(key);
        Logger.Log($"[SimAiPartner] {_partner.Label} takes {all[pick].OptionId} at the rest site ({(state == SimAskState.Answered ? "advisor" : "heal, no advice")}).");
        _partner.Send(new OptionIndexChosenMessage { type = OptionIndexType.RestSite, optionIndex = (uint)pick, location = Location });
    }

    /// <summary>The relics are awarded once every player has voted (a tie
    /// by rock-paper-scissors). The advisor scores them with its relic head;
    /// without it, the first.</summary>
    private void TickTreasure(Player me)
    {
        TreasureRoomRelicSynchronizer sync = RunManager.Instance.TreasureRoomRelicSynchronizer;
        string key = $"treasure:{Location}";
        if (sync?.CurrentRelics is not { Count: > 0 } relics || sync.GetPlayerVote(me).voteReceived || _sent.Contains(key))
        {
            return;
        }
        List<RelicModel> list = relics.ToList();
        (SimAskState state, int? index) = Choices.Ask(me, key, NonCombatBridgeProtocol.TreasureState,
            () => SimAiPartnerChoices.RelicState(me, list, NonCombatBridgeProtocol.TreasureState, "treasure"));
        if (state == SimAskState.Pending)
        {
            return;
        }
        int pick = state == SimAskState.Answered && index is int i && i < list.Count ? i : 0;
        Once(key);
        Logger.Log($"[SimAiPartner] {_partner.Label} votes for treasure relic {list[pick].Id.Entry} ({(state == SimAskState.Answered ? "advisor" : "first")}).");
        SendNetAction(new NetPickRelicAction { relicIndex = pick });
    }

    private static readonly FieldInfo? ReadyPlayersField =
        AccessTools.Field(typeof(ActChangeSynchronizer), "_readyPlayers");

    /// <summary>The host has voted to move on and we have not (private
    /// per-slot flags: IsWaitingForOtherPlayers is true whenever anyone
    /// else is not ready, i.e. all act long).</summary>
    private void TickActChange()
    {
        ActChangeSynchronizer sync = RunManager.Instance.ActChangeSynchronizer;
        RunState? state = RunManager.Instance.DebugOnlyGetState();
        Player? host = HostPlayer;
        if (sync == null || state == null || host == null || Me is not { } me ||
            ReadyPlayersField?.GetValue(sync) is not List<bool> ready)
        {
            return;
        }
        int hostSlot = state.GetPlayerSlotIndex(host);
        int mySlot = state.GetPlayerSlotIndex(me);
        if (hostSlot >= ready.Count || mySlot >= ready.Count || !ready[hostSlot] || ready[mySlot] ||
            !Once($"act:{state.CurrentActIndex}"))
        {
            return;
        }
        Logger.Log($"[SimAiPartner] {_partner.Label} votes to move on from act {state.CurrentActIndex}.");
        var vote = new NetVoteToMoveToNextActAction();
        // v0.107.1 has no currentActIndex (one DLL ships for both builds).
        object boxed = vote;
        AccessTools.Field(typeof(NetVoteToMoveToNextActAction), "currentActIndex")?.SetValue(boxed, state.CurrentActIndex);
        SendNetAction((INetAction)boxed);
    }

    /// <summary>The host moves once every player has voted: copy its vote.</summary>
    private void TickMapVote(Player me)
    {
        MapSelectionSynchronizer sync = RunManager.Instance.MapSelectionSynchronizer;
        Player? host = HostPlayer;
        if (sync == null || host == null || sync.GetVote(me) != null || sync.GetVote(host) is not MapVote hostVote)
        {
            return;
        }
        if (!Once($"map:{hostVote.mapGenerationCount}:{hostVote.coord}:{Location}"))
        {
            return;
        }
        Logger.Log($"[SimAiPartner] {_partner.Label} votes for map node {hostVote.coord} (the host's).");
        SendNetAction(new NetVoteForMapCoordAction { source = Location.mapLocation, destination = hostVote });
    }

    private static Player? HostPlayer =>
        SimAiPartnerSession.Active?.HostNetId is ulong id ? RunManager.Instance.DebugOnlyGetState()?.GetPlayer(id) : null;

    private static RunLocation Location => RunManager.Instance.RunLocationTargetedBuffer.CurrentLocation;

    /// <summary>Each peer enqueues its own ReadyToBeginEnemyTurnAction once
    /// the side has ended its turn (CombatManager, after phase one); the
    /// enemy turn waits for all of them.</summary>
    public void OnHostReadyForEnemyTurn()
    {
        if (Me is not { } me || !Once($"ready:{me.PlayerCombatState?.TurnNumber}:{Location}"))
        {
            return;
        }
        Logger.Log($"[SimAiPartner] {_partner.Label} is ready for the enemy turn.");
        SendAction(new ReadyToBeginEnemyTurnAction(me));
    }

    // ---- events ----------------------------------------------------------------

    private static readonly FieldInfo? CanonicalEventField =
        AccessTools.Field(typeof(EventSynchronizer), "_canonicalEvent");
    private static readonly FieldInfo? PageIndexField =
        AccessTools.Field(typeof(EventSynchronizer), "_pageIndex");

    /// <summary>
    /// An event room: non-shared events (Neow, most events) run one
    /// EventModel per player, and each peer sends the option its player
    /// chose (OptionIndexChosenMessage, EventSynchronizer.ChooseLocalOption);
    /// shared ones are a vote per page (VotedForSharedEventOptionMessage).
    /// The spike takes the first unlocked option, or the host's vote.
    /// </summary>
    private void TickEvent(Player me)
    {
        EventSynchronizer sync = RunManager.Instance.EventSynchronizer;
        if (sync == null || CanonicalEventField?.GetValue(sync) is not EventModel canonical)
        {
            return;
        }
        RunLocation location = RunManager.Instance.RunLocationTargetedBuffer.CurrentLocation;
        if (canonical.IsShared)
        {
            if (sync.GetPlayerVote(me).HasValue)
            {
                return;
            }
            uint page = PageIndexField?.GetValue(sync) is uint p ? p : 0u;
            // Each player has a mutable copy; the canonical model throws on
            // CurrentOptions (AssertMutable).
            IReadOnlyList<EventOption> shared = sync.GetEventForPlayer(me).CurrentOptions;
            Player? host = HostPlayer;
            uint? hostVote = host != null ? sync.GetPlayerVote(host) : null;
            int pick = hostVote.HasValue && hostVote.Value < shared.Count ? (int)hostVote.Value : FirstUnlocked(shared);
            if (pick < 0 || !Once($"vote:{location}:{canonical.Id.Entry}:{page}"))
            {
                return;
            }
            Logger.Log($"[SimAiPartner] {_partner.Label} votes option {pick} ({shared[pick].TextKey}) on shared event {canonical.Id.Entry} page {page}.");
            _partner.Send(new VotedForSharedEventOptionMessage { optionIndex = (uint)pick, pageIndex = page, location = location });
            return;
        }
        EventModel mine = sync.GetEventForPlayer(me);
        if (mine.IsFinished)
        {
            return;
        }
        IReadOnlyList<EventOption> options = mine.CurrentOptions;
        int first = FirstUnlocked(options);
        if (first < 0)
        {
            return;
        }
        // One message per page: the page's options change once the host has
        // run the choice.
        string key = $"event:{location}:{mine.Id.Entry}:" + string.Join("|", options.Select(o => o.TextKey));
        if (_sent.Contains(key))
        {
            return;
        }
        (SimAskState state, int? asked) = Choices.Ask(me, key, NonCombatBridgeProtocol.EventState, () =>
        {
            var s = new Dictionary<string, object>
            {
                [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.EventState,
                ["options"] = EventOptionRows(me, mine, options),
            };
            SimStateExtractor.ApplyEnvelope(s, me.RunState as RunState, me);
            return s;
        });
        if (state == SimAskState.Pending)
        {
            return;
        }
        int index = state == SimAskState.Answered && asked is int i && i < options.Count && !options[i].IsLocked
            ? i
            : SafeFallback(me, options, first);
        Once(key);
        Logger.Log($"[SimAiPartner] {_partner.Label} chooses option {index} ({options[index].TextKey}) in {mine.Id.Entry}" +
                   $" ({(state == SimAskState.Answered ? "advisor" : "no advice")}).");
        _partner.Send(new OptionIndexChosenMessage { type = OptionIndexType.Event, optionIndex = (uint)index, location = location });
    }

    /// <summary>The event's options in the wire shape of the human's lane
    /// (SimEventRoomHandler.EventOptions), indexed by their place in
    /// CurrentOptions, locked ones sent as disabled.</summary>
    private static List<Dictionary<string, object?>> EventOptionRows(Player me, EventModel ev, IReadOnlyList<EventOption> options) =>
        options.Select((o, i) =>
        {
            // The event's variables (MaxHp, Heal, ...) go onto the option's
            // strings before they are formatted, as NEventOptionButton does;
            // formatted bare, the game logs a localization error per selector.
            ev.DynamicVars.AddTo(o.Description);
            ev.DynamicVars.AddTo(o.Title);
            return o;
        }).Select((o, i) => new Dictionary<string, object?>
        {
            ["index"] = i,
            ["id"] = NonCombatBridgeProtocol.EventChoiceAction,
            ["action"] = NonCombatBridgeProtocol.EventChoiceAction,
            ["label"] = o.Title.GetFormattedText(),
            ["description"] = o.Description.GetFormattedText(),
            ["enabled"] = !o.IsLocked,
            ["event_id"] = ev.Id.Entry,
            ["text_key"] = o.TextKey ?? "",
            ["relic_id"] = o.Relic?.Id.Entry ?? "",
            ["will_kill_player"] = WillKill(o, me),
        }).ToList();

    /// <summary>The game's own lethality verdict for an option (null: it
    /// threw, i.e. unknown).</summary>
    private static bool? WillKill(EventOption option, Player me)
    {
        try
        {
            return option.WillKillPlayer?.Invoke(me) ?? false;
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>Without advice: the first unlocked option the game does not
    /// call lethal (a partner revived at 1 HP used to walk into an HP-cost
    /// option and die outside combat), else the first.</summary>
    private static int SafeFallback(Player me, IReadOnlyList<EventOption> options, int first)
    {
        for (int i = 0; i < options.Count; i++)
        {
            if (!options[i].IsLocked && WillKill(options[i], me) == false)
            {
                return i;
            }
        }
        return first;
    }

    private static int FirstUnlocked(IReadOnlyList<EventOption> options)
    {
        for (int i = 0; i < options.Count; i++)
        {
            if (!options[i].IsLocked)
            {
                return i;
            }
        }
        return -1;
    }

    // ---- combat --------------------------------------------------------------

    private void TickCombat(Player me)
    {
        if (me.Creature == null || !me.Creature.IsAlive || me.PlayerCombatState is not { } pcs)
        {
            Waiting = PartnerWait.None;
            return;
        }
        if (pcs.Phase != PlayerTurnPhase.Play || CombatManager.Instance.IsPlayerReadyToEndTurn(me))
        {
            Waiting = PartnerWait.None;
            return;
        }
        int turn = pcs.TurnNumber;
        if (turn != _turnSeen)
        {
            _turnSeen = turn;
            _playsThisTurn = 0;
            _pendingPlayKey = null;
        }
        List<CardModel> hand = PileType.Hand.GetPile(me).Cards.ToList();
        string handKey = $"{turn}:{pcs.Energy}:" + string.Join(",", hand.Select(c => c.Id.Entry));
        if (_pendingPlayKey != null)
        {
            // Wait for the last play to change the hand or energy.
            if (_pendingPlayKey == handKey && DateTime.UtcNow - _pendingSince < PlayTimeout)
            {
                return;
            }
            _pendingPlayKey = null;
        }
        // One partner play at a time, and only into an idle queue: a play
        // queued behind another that kills its target is Canceled when it
        // runs: a wasted card and a game ERROR line (GameAction.Execute; its
        // Sentry capture is off in modded sessions,
        // SentryService.DisableSentryIfModded). Two partners at one enemy
        // hit that often.
        if (!HostQueueIdle() || DateTime.UtcNow - _lastPartnerPlayAt < PartnerPlayGap)
        {
            return;
        }
        if (!HumansHaveGoneFirst(me, turn))
        {
            return;
        }
        if (Waiting is PartnerWait.FirstPlay or PartnerWait.EndTurn)
        {
            // The humans went first; a setup hold below sets its own.
            Waiting = PartnerWait.None;
        }
        if (_playsThisTurn >= MaxPlaysPerTurn)
        {
            EndTurn(me, turn);
            return;
        }
        if (TryAdvisedMove(me, pcs, hand, turn, handKey))
        {
            return;
        }
        foreach (CardModel card in hand)
        {
            if (!card.CanPlay())
            {
                continue;
            }
            Creature? target = SimCombatHandler.ResolveTarget(card, -1);
            if (NeedsTarget(card) && target == null)
            {
                continue;
            }
            if (!card.CanPlayTargeting(target))
            {
                continue;
            }
            Play(card, target, turn, handKey, "first legal");
            return;
        }
        EndTurn(me, turn);
    }

    private void Play(CardModel card, Creature? target, int turn, string handKey, string why)
    {
        Logger.Log($"[SimAiPartner] {_partner.Label} plays {card.Id.Entry}" +
                   (target != null ? $" at {target.Name}" : "") + $" (turn {turn}; {why}).");
        int targetIndex = TargetIndexOf(target);
        int? allyIndex = AllyIndexOf(card.Owner, target);
        SendAction(new PlayCardAction(card, target));
        ReportAiPlay(card, null, targetIndex, allyIndex);
        _lastPartnerPlayAt = DateTime.UtcNow;
        _pendingPlayKey = handKey;
        _pendingSince = DateTime.UtcNow;
        _playsThisTurn++;
    }

    private void EndTurn(Player me, int turn)
    {
        // Re-send when still not ready a while later: a prompt for us in our
        // turn un-ends it (CardSelectCmd.UndoEndTurnIfNecessary).
        if (_endTurnSentFor != turn || DateTime.UtcNow - _endTurnSentAt > EndTurnRetry)
        {
            _endTurnSentFor = turn;
            _endTurnSentAt = DateTime.UtcNow;
            Logger.Log($"[SimAiPartner] {_partner.Label} ends turn {turn}.");
            SendAction(new EndPlayerTurnAction(me, turn));
        }
    }

    // ---- combat advice ----------------------------------------------------------

    /// <summary>How long a combat request may stay unanswered before the
    /// partner plays that board without advice (the first request after the
    /// advisor connects also pays for its warm-up).</summary>
    private static readonly TimeSpan AdviceTimeout = TimeSpan.FromSeconds(30);

    /// <summary>The longest the partner holds a move for a teammate's setup
    /// play (the advice's <c>wait_for_ally</c>): for a human, the settings
    /// page's "holds for your setup play" (default 10 s, up to 60; 0 =
    /// never), since a person reads the board and thinks, and "Partner go"
    /// ends it early; for another AI partner, briefly (PartnerCharacters).</summary>
    private static TimeSpan SetupHold(bool humanStillToAct) =>
        TimeSpan.FromSeconds(PartnerCharacters.HoldSeconds(SpireMentorModConfig.AiPartnerSetupHoldSeconds, humanStillToAct));

    private int _paceLoggedFor = -1;

    /// <summary>The turn "Partner: go" released (<see cref="Go"/>): no more
    /// waiting for the humans or holding for their setup until it ends.</summary>
    private int _goTurn = -1;

    /// <summary>What the partner is waiting for in its combat turn, for the
    /// toolbar's "Partner go" (main thread, like the tick that sets it).</summary>
    public PartnerWait Waiting { get; private set; }

    /// <summary>Whole seconds left of a setup hold for a human
    /// (<see cref="PartnerWait.SetupHold"/>), else 0.</summary>
    public int HoldSecondsLeft =>
        Waiting == PartnerWait.SetupHold ? (int)Math.Ceiling(Math.Max(0, (_heldUntil - DateTime.UtcNow).TotalSeconds)) : 0;

    /// <summary>"Partner go" (toolbar, Shift+F6): play the rest of this turn
    /// now. False when the partner was not waiting (nothing to release).</summary>
    public bool Go()
    {
        if (Waiting == PartnerWait.None)
        {
            return false;
        }
        Logger.Log($"[SimAiPartner] {_partner.Label}: go (was waiting: {Waiting}; turn {_turnSeen}).");
        _goTurn = _turnSeen;
        Waiting = PartnerWait.None;
        return true;
    }

    /// <summary>
    /// The settings' "let me go first" (read live): with "waits until you end
    /// your turn", the partner starts once every living human player has
    /// ended theirs; with "waits for your first play", once each has played a
    /// card or ended. Off by default: both sides play the same turn, as
    /// co-op does. An AI partner never waits for another AI partner.
    /// </summary>
    private bool HumansHaveGoneFirst(Player me, int turn)
    {
        string? pace = SimMp.AiPartnerPaceOverride;
        bool forEnd = pace != null ? pace == "end-turn" : SpireMentorModConfig.AiPartnerWaitForEndTurn;
        bool forPlay = pace != null ? pace == "first-play" : SpireMentorModConfig.AiPartnerWaitForFirstPlay;
        if ((!forEnd && !forPlay) || _goTurn == turn || CombatManager.Instance.DebugOnlyGetState() is not { } combat)
        {
            return true;
        }
        foreach (Player p in combat.Players)
        {
            if (ReferenceEquals(p, me) || SimAiPartner.IsPartner(p.NetId) || p.Creature is not { IsAlive: true })
            {
                continue;
            }
            if (CombatManager.Instance.IsPlayerReadyToEndTurn(p))
            {
                continue;
            }
            if (!forEnd && SimCombatHandler.CountCardsPlayedThisTurn(combat, p.Creature).Sum() > 0)
            {
                continue;
            }
            if (_paceLoggedFor != turn)
            {
                _paceLoggedFor = turn;
                Logger.Log($"[SimAiPartner] {_partner.Label} waits for the human player's " +
                           (forEnd ? "end turn" : "first play") + $" (turn {turn}).");
            }
            Waiting = forEnd ? PartnerWait.EndTurn : PartnerWait.FirstPlay;
            return false;
        }
        return true;
    }

    private sealed record AdviceReply(string Signature, SimAdvice Advice, long StateId);

    private readonly object _replyLock = new();
    private AdviceReply? _reply;
    private string? _inflightSignature;
    private DateTime _inflightAt;
    private string? _gaveUpSignature;
    private string? _heldSignature;
    private DateTime _heldSince;
    private DateTime _heldUntil;
    private long _boardStateId = -1;

    /// <summary>
    /// The advisor's move for this board: asks for it (once per board), waits
    /// for the reply, plays it. True when the partner acted or is waiting;
    /// false hands the board to the first-legal fallback (no advisor, a
    /// timeout, an unusable reply).
    /// </summary>
    private bool TryAdvisedMove(Player me, PlayerCombatState pcs, List<CardModel> hand, int turn, string handKey)
    {
        SimAiPartnerAdvisor? advisor = _partner.Advisor;
        if (advisor == null || !advisor.Connected)
        {
            return false;
        }
        string sig = BoardSignature(me, pcs, hand);
        if (sig == _gaveUpSignature)
        {
            return false;
        }
        AdviceReply? reply;
        lock (_replyLock)
        {
            reply = _reply;
        }
        if (reply != null && reply.Signature == sig)
        {
            bool humanToAct = AllyStillToAct(me, humansOnly: true);
            TimeSpan hold = SetupHold(humanToAct);
            if (reply.Advice.WaitForAllyMs > 0 && hold > TimeSpan.Zero && _goTurn != turn && AllyStillToAct(me))
            {
                // A teammate can make this move better (their Vulnerable on
                // our target, their block on us): hold it for the configured
                // time (the advisor's own max_ms is sized for its autoplay;
                // a human thinking needs the player's setting). Their play
                // changes the board, which asks again.
                if (_heldSignature != sig)
                {
                    _heldSignature = sig;
                    _heldSince = DateTime.UtcNow;
                    Logger.Log($"[SimAiPartner] {_partner.Label} holds its move up to {hold.TotalSeconds:0.0} s for a " +
                               (humanToAct ? "human" : "partner") + " teammate's setup play.");
                }
                // From the hold's start, by who is still to act now: a human
                // ending their turn mid-hold leaves only the short AI hold.
                _heldUntil = _heldSince + hold;
                TimeSpan left = _heldUntil - DateTime.UtcNow;
                if (left > TimeSpan.Zero)
                {
                    // Go only releases a hold for a human: an AI teammate's
                    // short hold ends by itself.
                    Waiting = humanToAct ? PartnerWait.SetupHold : PartnerWait.None;
                    return true;
                }
            }
            Waiting = PartnerWait.None;
            lock (_replyLock)
            {
                _reply = null;
            }
            _inflightSignature = null;
            _boardStateId = reply.StateId;
            if (ExecuteAdvice(me, hand, turn, handKey, reply.Advice))
            {
                return true;
            }
            _gaveUpSignature = sig;
            return false;
        }
        if (_inflightSignature == sig)
        {
            if (DateTime.UtcNow - _inflightAt < AdviceTimeout)
            {
                return true;
            }
            Logger.Log($"[SimAiPartner] {_partner.Label}: no advice within {AdviceTimeout.TotalSeconds:0} s; playing this board without it.");
            _gaveUpSignature = sig;
            _inflightSignature = null;
            return false;
        }
        Dictionary<string, object>? state = BuildAsMe(me, () => SimCombatHandler.BuildCombatState(me));
        if (state == null)
        {
            return false;
        }
        state["turn"] = turn;
        state["type"] = "advice_request";
        state["advice_for"] = "combat_action";
        long stateId = state.TryGetValue("state_id", out object? sidObj) && sidObj is long sid ? sid : -1;
        _inflightSignature = sig;
        _inflightAt = DateTime.UtcNow;
        lock (_replyLock)
        {
            _reply = null;
        }
        if (!advisor.Request(state, root => OnAdviceReply(sig, stateId, root)))
        {
            _inflightSignature = null;
            return false;
        }
        return true;
    }

    /// <summary>Socket thread: parse, park for the next tick.</summary>
    private void OnAdviceReply(string signature, long stateId, JsonElement root)
    {
        try
        {
            if (!root.TryGetProperty("advice", out JsonElement adviceEl))
            {
                return;
            }
            SimAdvice advice = SimAdvice.Parse(adviceEl);
            lock (_replyLock)
            {
                _reply = new AdviceReply(signature, advice, stateId);
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartner] {_partner.Label}: unreadable advice: {ex.Message}");
        }
    }

    private bool ExecuteAdvice(Player me, List<CardModel> hand, int turn, string handKey, SimAdvice advice)
    {
        SimBestAction? best = advice.Best;
        if (best == null)
        {
            return false;
        }
        switch (best.Action)
        {
            case "end_turn":
                EndTurn(me, turn);
                return true;
            case "play":
            {
                if (best.CardIndex < 0 || best.CardIndex >= hand.Count)
                {
                    return false;
                }
                CardModel card = hand[best.CardIndex];
                if (!card.CanPlay())
                {
                    return false;
                }
                Creature? target = SimCombatHandler.ResolveTarget(card, best.TargetIndex, best.AllyIndex);
                if ((NeedsTarget(card) && target == null) || !card.CanPlayTargeting(target))
                {
                    return false;
                }
                Play(card, target, turn, handKey, $"advisor{(advice.PWin is double p ? $", p_win {p:0.00}" : "")}");
                return true;
            }
            case "potion":
            {
                PotionModel? potion = SimCombatHandler.SafeGetPotionAtSlot(me, best.Slot);
                if (potion == null)
                {
                    return false;
                }
                Creature? target = SimCombatHandler.ResolvePotionTarget(me, potion, best.TargetIndex, best.AllyIndex);
                Logger.Log($"[SimAiPartner] {_partner.Label} drinks {potion.Id.Entry}" +
                           (target != null ? $" at {target.Name}" : "") + $" (turn {turn}; advisor).");
                SendAction(new UsePotionAction(potion, target, CombatManager.Instance.IsInProgress));
                ReportAiPlay(null, potion, TargetIndexOf(target), AllyIndexOf(me, target));
                _lastPartnerPlayAt = DateTime.UtcNow;
                _pendingPlayKey = handKey + ":potion" + best.Slot;
                _pendingSince = DateTime.UtcNow;
                _playsThisTurn++;
                return true;
            }
            default:
                return false;
        }
    }

    /// <summary>
    /// Builds a payload with the game's "me" pointed at this partner. C#
    /// reads the local player in places the payload builders cannot pass a
    /// player to -- AttackIntent.GetSingleDamage computes an intent's damage
    /// against LocalContext.GetMe (the host's Vulnerable, not ours) -- and
    /// Hook.ModifyDamage's signature differs between the builds, so it cannot
    /// be called directly. The build is synchronous on the main thread;
    /// nothing else runs before the id is restored.
    /// </summary>
    internal Dictionary<string, object>? BuildAsMe(Player me, Func<Dictionary<string, object>?> build)
    {
        ulong? previous = LocalContext.NetId;
        try
        {
            LocalContext.NetId = me.NetId;
            return build();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartner] {_partner.Label}: could not build the board: {ex.Message}");
            return null;
        }
        finally
        {
            LocalContext.NetId = previous;
        }
    }

    /// <summary>What a reply must still match to be played: our turn,
    /// energy, hand, belt, HP and block, and every enemy's HP, block and
    /// powers. A teammate's play changes it; a stale reply is dropped and
    /// the board asked again.</summary>
    private static string BoardSignature(Player me, PlayerCombatState pcs, List<CardModel> hand)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append(pcs.TurnNumber).Append('|').Append(pcs.Energy).Append('|')
          .Append(me.Creature?.CurrentHp).Append('/').Append(me.Creature?.Block).Append('|');
        foreach (CardModel c in hand)
        {
            sb.Append(c.Id.Entry).Append(c.CurrentUpgradeLevel > 0 ? "+" : "").Append(',');
        }
        sb.Append('|');
        foreach (PotionModel? p in me.PotionSlots)
        {
            sb.Append(p?.Id.Entry ?? "-").Append(',');
        }
        sb.Append('|');
        if (CombatManager.Instance.DebugOnlyGetState() is { } combat)
        {
            foreach (Creature e in combat.Enemies)
            {
                sb.Append(e.CombatId).Append(':').Append(e.CurrentHp).Append('/').Append(e.Block)
                  .Append('/').Append(e.Powers.Count).Append(',');
            }
            foreach (Player p in combat.Players)
            {
                if (!ReferenceEquals(p, me))
                {
                    sb.Append("A").Append(p.Creature?.Block).Append('/').Append(p.Creature?.CurrentHp).Append(',');
                }
            }
        }
        return sb.ToString();
    }

    /// <summary>Some living teammate has not ended its turn yet; with
    /// <paramref name="humansOnly"/>, a human one (not an AI partner).</summary>
    private static bool AllyStillToAct(Player me, bool humansOnly = false)
    {
        if (CombatManager.Instance.DebugOnlyGetState() is not { } combat)
        {
            return false;
        }
        return combat.Players.Any(p => !ReferenceEquals(p, me) && p.Creature is { IsAlive: true } &&
                                       (!humansOnly || !SimAiPartner.IsPartner(p.NetId)) &&
                                       !CombatManager.Instance.IsPlayerReadyToEndTurn(p));
    }

    private static int TargetIndexOf(Creature? target)
    {
        if (target == null || CombatManager.Instance.DebugOnlyGetState() is not { } combat)
        {
            return -1;
        }
        for (int i = 0; i < combat.Enemies.Count; i++)
        {
            if (ReferenceEquals(combat.Enemies[i], target))
            {
                return i;
            }
        }
        return -1;
    }

    /// <summary>A teammate target's index in our board's <c>allies[]</c>
    /// (the other players in run order), else null.</summary>
    private static int? AllyIndexOf(Player me, Creature? target)
    {
        if (target == null || !target.IsPlayer || me.RunState is not { } run)
        {
            return null;
        }
        int i = 0;
        foreach (Player p in run.Players)
        {
            if (ReferenceEquals(p, me))
            {
                continue;
            }
            if (ReferenceEquals(p.Creature, target))
            {
                return i;
            }
            i++;
        }
        return null;
    }

    /// <summary>
    /// The machine form of our play for the advisor's combat mirror
    /// (PROTOCOL.md human_action, `ai_play`), as the human's autoplay sends:
    /// without it the mirror can only re-seed from each new board.
    /// </summary>
    private void ReportAiPlay(CardModel? card, PotionModel? potion, int targetIndex, int? allyIndex)
    {
        SimAiPartnerAdvisor? advisor = _partner.Advisor;
        if (advisor == null || !advisor.Connected)
        {
            return;
        }
        var msg = new Dictionary<string, object?>
        {
            ["type"] = "human_action",
            ["state_id"] = _boardStateId >= 0 ? _boardStateId : null,
            ["decision_type"] = "combat_action",
            ["action"] = "ai_play",
            ["detail"] = card != null ? $"AI: {card.Id.Entry}" : $"AI: {potion?.Id.Entry}",
            ["card"] = card == null
                ? null
                : new Dictionary<string, object?> { ["id"] = card.Id.Entry, ["upgraded"] = card.CurrentUpgradeLevel > 0 },
            ["potion"] = potion == null ? null : new Dictionary<string, object?> { ["id"] = potion.Id.Entry },
            ["target_index"] = targetIndex >= 0 ? targetIndex : null,
        };
        if (allyIndex != null)
        {
            msg["ally_index"] = allyIndex;
        }
        advisor.Report(msg);
    }

    /// <summary>Across all partners: the last play sent. The request needs a
    /// round trip (a frame or two) before it shows in the host's queue.</summary>
    private static DateTime _lastPartnerPlayAt;
    private static readonly TimeSpan PartnerPlayGap = TimeSpan.FromMilliseconds(400);

    private static bool HostQueueIdle()
    {
        RunManager run = RunManager.Instance;
        return run.ActionQueueSet is { IsEmpty: true } && run.ActionExecutor is { CurrentlyRunningAction: null };
    }

    private static bool NeedsTarget(CardModel card) =>
        card.TargetType == TargetType.AnyEnemy || card.TargetType == TargetType.AnyAlly;

    /// <summary>A GameAction built against the host's state for our Player,
    /// sent the way a client asks the host to enqueue it
    /// (ActionQueueSynchronizer.RequestEnqueue, client branch).</summary>
    public void SendAction(GameAction action) => SendNetAction(action.ToNetAction());

    private void SendNetAction(INetAction action)
    {
        _partner.Send(new RequestEnqueueActionMessage { action = action, location = Location });
    }

    /// <summary>Combat start: every peer sends its own player's data, and
    /// the host's WaitForSync blocks until every player's has arrived
    /// (CombatStateSynchronizer.StartSync).</summary>
    public void OnCombatSyncStarted()
    {
        Player? me = Me;
        if (me == null)
        {
            return;
        }
        Logger.Log($"[SimAiPartner] {_partner.Label} sends its combat sync data.");
        _partner.Send(new SyncPlayerDataMessage { player = me.ToSerializable() });
        SendPeerInput();
    }

    /// <summary>
    /// A real client streams its cursor and screen (PeerInputMessage) from the
    /// start, so the host's PeerInputSynchronizer already has its state when
    /// a screen asks. For a peer that never sent one the host creates it on
    /// first use, inside the treasure room's own hand setup, which then adds
    /// that player's hand twice (an ERROR line). One default message per room
    /// (the synchronizer is the run's, not the lobby's) prevents it. Default
    /// fields only: v0.111.0 has two more than v0.107.1.
    /// </summary>
    public void SendPeerInput()
    {
        _partner.Send(new PeerInputMessage());
    }

    /// <summary>True the first time for a key: one message per decision.</summary>
    public bool Once(string key) => _sent.Add(key);
}

internal static class SimAiPartnerPatches
{
    public static void Install(Harmony patcher)
    {
        Patch(patcher, AccessTools.Method(typeof(CombatStateSynchronizer), nameof(CombatStateSynchronizer.StartSync)),
            postfix: nameof(StartSyncPostfix));
        Patch(patcher, AccessTools.Method(typeof(ActionQueueSynchronizer), nameof(ActionQueueSynchronizer.RequestEnqueue)),
            postfix: nameof(RequestEnqueuePostfix));
        Patch(patcher, AccessTools.Method(typeof(PlayerChoiceSynchronizer), nameof(PlayerChoiceSynchronizer.WaitForRemoteChoice)),
            prefix: nameof(WaitForRemoteChoicePrefix));
        Patch(patcher, AccessTools.Method(typeof(RewardsSetSynchronizer), nameof(RewardsSetSynchronizer.BeginRewardsSet)),
            postfix: nameof(BeginRewardsSetPostfix));
        foreach (MethodInfo m in typeof(CardSelectCmd).GetMethods(BindingFlags.Public | BindingFlags.Static)
                     .Concat(typeof(RelicSelectCmd).GetMethods(BindingFlags.Public | BindingFlags.Static))
                     .Where(m => m.Name.StartsWith("From", StringComparison.Ordinal) && !m.IsGenericMethodDefinition))
        {
            Patch(patcher, m, prefix: nameof(PromptPrefix), quiet: true);
        }
        Patch(patcher, AccessTools.Method(typeof(CardReward), "OnSelect"), prefix: nameof(CardRewardPrefix));
        InstallHookResumes(patcher);
    }

    // ---- hook choices: the host resumes for its partners ---------------------
    //
    // A choice inside a hook or a shared-queue action (Seeker Strike's pick,
    // a relic's "choose a card") pauses a game action that only its OWNER
    // resumes: the owner's peer sends RequestEnqueueHookActionMessage /
    // RequestResumeActionAfterPlayerChoiceMessage, every other peer waits
    // (HookPlayerChoiceContext, GameActionPlayerChoiceContext: `OwnerId ==
    // local player`). A partner runs no simulation, so nothing on its side
    // would ever see the pause. The host does it for the partner instead,
    // through the same calls the owner's peer makes (the host-side
    // RequestEnqueueHookAction / RequestResumeActionAfterPlayerChoice
    // enqueue / resume and broadcast it). The game's tests do this for every
    // remote player with TestFlags.ShouldSendResumeForRemotePlayers; that
    // would also resume for a human client's choices (plan, phase 4), so
    // these patches do it for a partner's actions only. The same code in both
    // builds (v0.107.1 has no TestFlags).

    private static readonly FieldInfo? HookContextLocalPlayer =
        AccessTools.Field(typeof(HookPlayerChoiceContext), "_localPlayerId");

    private static void InstallHookResumes(Harmony patcher)
    {
        if (HookContextLocalPlayer == null)
        {
            Logger.Log("  patch skipped: SimAiPartner hook choices (HookPlayerChoiceContext._localPlayerId not found)");
        }
        else
        {
            Patch(patcher, AccessTools.Method(typeof(HookPlayerChoiceContext), nameof(HookPlayerChoiceContext.SignalPlayerChoiceBegun)),
                prefix: nameof(HookChoiceBegunPrefix));
        }
        Patch(patcher, AccessTools.Method(typeof(GameActionPlayerChoiceContext), nameof(GameActionPlayerChoiceContext.SignalPlayerChoiceEnded)),
            prefix: nameof(ActionChoiceEndedPrefix));
    }

    /// <summary>The context's game action is the chooser's, and the context
    /// enqueues / resumes it when its "local player" is the owner: make the
    /// partner the local player of its own choice's context. Runs before
    /// the async body, which reads the field after its awaits. The action's
    /// owner is the chooser (v0.111.0's argument) or the context's Owner
    /// (v0.107.1 takes no chooser).</summary>
    private static void HookChoiceBegunPrefix(HookPlayerChoiceContext __instance, object[] __args)
    {
        try
        {
            Player? chooser = __args.OfType<Player>().FirstOrDefault() ?? __instance.Owner;
            if (chooser != null && PartnerFor(chooser) != null)
            {
                HookContextLocalPlayer!.SetValue(__instance, chooser.NetId);
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartner] hook choice: {ex.Message}");
        }
    }

    /// <summary>A shared-queue action's choice ends: its owner's peer
    /// requests the resume (`Action.OwnerId == LocalContext.NetId`), so the
    /// host does for a partner's. The original then only waits.</summary>
    private static void ActionChoiceEndedPrefix(GameActionPlayerChoiceContext __instance)
    {
        try
        {
            GameAction action = __instance.Action;
            if (!SimAiPartner.All.Any(p => p.Joined && p.NetId == action.OwnerId))
            {
                return;
            }
            Logger.Log($"[SimAiPartner] resuming {action} after its owner's choice (partner {action.OwnerId}).");
            RunManager.Instance.ActionQueueSynchronizer.RequestResumeActionAfterPlayerChoice(action);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartner] action choice resume: {ex.Message}");
        }
    }

    private static void Patch(Harmony patcher, MethodBase? target, string? prefix = null, string? postfix = null, bool quiet = false)
    {
        if (target == null)
        {
            Logger.Log($"  patch skipped: SimAiPartner {prefix ?? postfix} (target not found)");
            return;
        }
        try
        {
            patcher.Patch(target,
                prefix: prefix != null ? new HarmonyMethod(typeof(SimAiPartnerPatches), prefix) : null,
                postfix: postfix != null ? new HarmonyMethod(typeof(SimAiPartnerPatches), postfix) : null);
            if (!quiet)
            {
                Logger.Log($"  patch applied: SimAiPartner {target.DeclaringType?.Name}.{target.Name}");
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"  patch skipped: SimAiPartner {target.DeclaringType?.Name}.{target.Name} ({ex.Message})");
        }
    }

    private static SimAiPartner? PartnerFor(Player? player) =>
        player == null ? null : SimAiPartner.All.FirstOrDefault(p => p.Joined && p.NetId == player.NetId);

    private static void RequestEnqueuePostfix(GameAction action)
    {
        try
        {
            if (action is not ReadyToBeginEnemyTurnAction || SimAiPartner.IsPartner(action.OwnerId))
            {
                return;
            }
            foreach (SimAiPartner p in SimAiPartner.All.Where(p => p.Joined))
            {
                p.Brain.OnHostReadyForEnemyTurn();
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartner] ready-for-enemy-turn: {ex.Message}");
        }
    }

    private static void WaitForRemoteChoicePrefix(Player player, uint choiceId)
    {
        PartnerFor(player)?.Brain.OnRemoteChoiceAwaited(choiceId);
    }

    private static void BeginRewardsSetPostfix(RewardsSet set)
    {
        try
        {
            // An empty set (a rest site's, a treasure room's) completes by
            // itself; skipping it makes the host throw ("not currently
            // viewing any reward set").
            if (set.Rewards.Count > 0)
            {
                PartnerFor(set.Player)?.Brain.OnRewardsSetOffered(set);
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAiPartner] rewards set: {ex.Message}");
        }
    }

    /// <summary>Records which prompt family a partner's next remote choice
    /// answers (the args name the player; the deck-enchantment overload
    /// only names the cards).</summary>
    private static void PromptPrefix(MethodBase __originalMethod, object[] __args)
    {
        try
        {
            Player? player = __args.OfType<Player>().FirstOrDefault()
                             ?? (__args.OfType<IReadOnlyList<CardModel>>().FirstOrDefault()?.FirstOrDefault()?.Owner);
            if (PartnerFor(player) is { } p)
            {
                p.Brain.Choices.Prompt = new SimAiPartnerPrompt($"{__originalMethod.DeclaringType?.Name}.{__originalMethod.Name}", __args);
            }
        }
        catch (Exception)
        {
            // Observation only.
        }
    }

    private static void CardRewardPrefix(CardReward __instance)
    {
        try
        {
            if (PartnerFor(__instance.Player) is { } p)
            {
                p.Brain.Choices.Prompt = new SimAiPartnerPrompt("CardReward.OnSelect", new object?[] { __instance });
            }
        }
        catch (Exception)
        {
            // Observation only.
        }
    }

    private static void StartSyncPostfix()
    {
        foreach (SimAiPartner p in SimAiPartner.All)
        {
            if (!p.Joined)
            {
                continue;
            }
            try
            {
                p.Brain.OnCombatSyncStarted();
            }
            catch (Exception ex)
            {
                Logger.Log($"[SimAiPartner] {p.Label} combat sync: {ex.Message}");
            }
        }
    }
}

/// <summary>Why an AI partner is not playing its combat turn yet
/// (SimAiPartnerBrain.Waiting); "Partner go" releases any of them.</summary>
internal enum PartnerWait
{
    None,
    /// <summary>"waits for your first play": a human has not played a card.</summary>
    FirstPlay,
    /// <summary>"waits until you end your turn".</summary>
    EndTurn,
    /// <summary>Its best move gets better after a human's setup play
    /// (wait_for_ally); held up to "holds for your setup play".</summary>
    SetupHold,
}
