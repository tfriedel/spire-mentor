// SimCombatHandler.cs -- agent-driven combat room handler and the combat_action board builder.
// Originally ported from zhiyue/sts2-rl-agent (STS2BridgeMod); rewritten clean-room.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.AutoSlay;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Actions;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Afflictions;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

/// <summary>
/// Room handler for Monster / Elite / Boss rooms. Every combat decision is
/// delegated to the bridge agent: the handler serializes the board, waits
/// (without limit) for a play / end_turn / potion reply, applies it, lets the
/// engine settle, and repeats until the fight is over.
/// </summary>
public class SimCombatHandler : IRoomHandler, IHandler
{
    // Turn-loop bounds and cadence (tuned values; see SPEC 12.4).
    private const int TurnCap = 200;
    private const int ActionCapPerTurn = 50;
    private const int WatchdogRefreshEveryActions = 10;
    private const int FrameSettleMs = 16;
    private static readonly TimeSpan PlayPhaseWait = TimeSpan.FromSeconds(30);
    private static readonly TimeSpan CombatEndWait = TimeSpan.FromSeconds(30);

    private static readonly RoomType[] CombatRoomTypes =
    {
        RoomType.Monster,
        RoomType.Elite,
        RoomType.Boss,
    };

    public RoomType[] HandledTypes => CombatRoomTypes;

    public TimeSpan Timeout => TimeSpan.FromHours(24);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        Logger.Log("[SimCombat] Waiting for combat to start");
        await WaitHelper.Until(() => CombatManager.Instance.IsInProgress, ct,
            AutoSlayConfig.nodeWaitTimeout, "Combat never started");
        Logger.Log("[SimCombat] Combat started");

        Player player = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState())
            ?? throw new InvalidOperationException("SimCombat: no local player in the run state");

        for (int turn = 1; turn <= TurnCap && CombatManager.Instance.IsInProgress; turn++)
        {
            ct.ThrowIfCancellationRequested();

            await WaitHelper.Until(
                () => IsPlayerInPlayPhase(player) || !CombatManager.Instance.IsInProgress,
                ct, PlayPhaseWait, $"Player turn {turn} never reached the play phase");
            if (!CombatManager.Instance.IsInProgress)
                break;

            SimAutoSlayer.CurrentWatchdog?.Reset($"Combat turn {turn}");
            Logger.Log($"[SimCombat] Turn {turn}: awaiting agent decision");

            bool agentEndedTurn = await RunTurnDecisionsAsync(player, turn, ct);

            // Only reachable at the per-turn action cap: the agent never said
            // end_turn, but the turn has to finish somehow.
            if (!agentEndedTurn && IsPlayerInPlayPhase(player) && CombatManager.Instance.IsInProgress)
            {
                Logger.Log($"[SimCombat] Turn {turn}: {ActionCapPerTurn}-action cap reached, forcing end turn");
                SimAdvisor.EndTurnAsAi(player);
                await WaitForTurnEndAsync(player, ct);
            }
        }

        await WaitHelper.Until(() => !CombatManager.Instance.IsInProgress, ct,
            CombatEndWait, "Combat did not end");
        Logger.Log("[SimCombat] Combat finished");
    }

    /// <summary>
    /// Asks the agent for actions until it ends the turn, the action cap is
    /// hit, or the player drops out of the play phase. Returns whether the
    /// agent itself ended the turn. Rejected / errored replies still count as
    /// an action so a misbehaving agent cannot spin the turn forever.
    /// </summary>
    private async Task<bool> RunTurnDecisionsAsync(Player player, int turn, CancellationToken ct)
    {
        int actionsTaken = 0;
        while (actionsTaken < ActionCapPerTurn && IsPlayerInPlayPhase(player))
        {
            ct.ThrowIfCancellationRequested();

            if (actionsTaken > 0 && actionsTaken % WatchdogRefreshEveryActions == 0)
                SimAutoSlayer.CurrentWatchdog?.Reset($"Combat turn {turn}, played {actionsTaken} cards");

            (string stateJson, long stateId) = SerializeCombatState(player);
            string reply = await SimAgentGate.RequestActionAsync(stateJson, ct, "combat_action");
            bool endedTurn = await ExecuteAgentAction(reply, player, stateId, ct);

            await Task.Delay(FrameSettleMs, ct);
            if (endedTurn)
                return true;
            actionsTaken++;
        }
        return false;
    }

    /// <summary>
    /// The teammate an ally-target card or potion goes to (co-op only; such
    /// cards are unplayable alone). <paramref name="allyIndex"/> is an index
    /// into the board's <c>allies[]</c>, the other players in run order as
    /// <see cref="SerializeAllies"/> sends them, every one of them (GH #1017:
    /// with three or four players the advisor names one, the index of the
    /// engine's <c>PlayCardAtAlly</c>); it must be alive, unless only one
    /// teammate is. Negative: the first living teammate, the only one with
    /// two players, and where the engine aims an untargeted AnyAlly play
    /// (<c>CombatState::any_ally_card_target</c>).
    /// </summary>
    internal static Creature? TeammateTarget(Player player, int allyIndex = -1)
    {
        IRunState? runState = player.RunState;
        if (runState == null)
            return null;
        var players = runState.Players.ToList();
        var table = players
            .Select(p => (IsMe: ReferenceEquals(p, player), IsAlive: p.Creature?.IsAlive ?? false))
            .ToList();
        int i = TeammateTargetPick.Pick(table, allyIndex);
        return i < 0 ? null : players[i].Creature;
    }

    private static bool IsPlayerInPlayPhase(Player player) =>
        player.PlayerCombatState?.Phase == PlayerTurnPhase.Play;

    /// <summary>
    /// Target for a card play. AnyEnemy cards take an enemy: a non-negative
    /// index addresses the full enemy list (so indices match the serialized
    /// <c>enemies</c> array) and must land on a hittable enemy; a negative
    /// index means "first hittable enemy". Co-op's AnyAlly cards ignore it
    /// and take the teammate <paramref name="allyIndex"/> names in
    /// <c>allies[]</c> (<see cref="TeammateTarget"/>).
    /// </summary>
    internal static Creature? ResolveTarget(CardModel card, int targetIndex, int allyIndex = -1)
    {
        if (card.TargetType == TargetType.AnyAlly)
            return TeammateTarget(card.Owner, allyIndex);
        if (card.TargetType != TargetType.AnyEnemy)
            return null;

        ICombatState? combat = card.CombatState;
        if (combat == null || combat.Enemies.Count == 0)
            return null;

        if (targetIndex < 0)
            return combat.HittableEnemies.FirstOrDefault();

        if (targetIndex >= combat.Enemies.Count)
            return null;

        Creature candidate = combat.Enemies[targetIndex];
        return candidate.IsHittable ? candidate : null;
    }

    /// <summary>
    /// Applies one agent reply (PROTOCOL.md section 4, combat verbs). Returns
    /// true only when the reply ended the turn. Every rejection sends a
    /// structured error and returns false, so the turn loop re-serializes and
    /// asks again.
    /// </summary>
    private async Task<bool> ExecuteAgentAction(string json, Player player, long stateId, CancellationToken ct)
    {
        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        if (!SimProtocolHelpers.TryParse(json, runState, player, out JsonElement root) ||
            !SimProtocolHelpers.CheckStateId(root, stateId, runState, player))
        {
            return false;
        }

        try
        {
            string action = root.ValueKind == JsonValueKind.Object &&
                            root.TryGetProperty("action", out JsonElement actionEl) &&
                            actionEl.ValueKind == JsonValueKind.String
                ? actionEl.GetString() ?? ""
                : "";

            if (action.Equals("play", StringComparison.OrdinalIgnoreCase))
            {
                await ExecutePlayAsync(root, player, runState, ct);
                return false;
            }
            if (action.Equals("end_turn", StringComparison.OrdinalIgnoreCase))
            {
                Logger.Log("[SimCombat] Agent ended the turn");
                SimAdvisor.EndTurnAsAi(player);
                await WaitForTurnEndAsync(player, ct);
                return true;
            }
            if (action.Equals("potion", StringComparison.OrdinalIgnoreCase))
            {
                await ExecutePotionAsync(root, player, runState, ct);
                return false;
            }

            SimProtocolHelpers.SendError("unknown_action", $"Unknown combat action \"{action}\"", runState, player);
            return false;
        }
        // CLEANROOM2-Q6: cancellation (run teardown) propagates instead of
        // being reported to the agent as internal_error.
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            Logger.Log($"[SimCombat] Error applying agent action: {ex}");
            SimProtocolHelpers.SendError("internal_error", ex.Message, runState, player);
            return false;
        }
    }

    /// <summary>The <c>play</c> verb: card_index into the hand, optional target_index.</summary>
    private static async Task ExecutePlayAsync(JsonElement root, Player player, RunState? runState, CancellationToken ct)
    {
        if (!TryReadInt(root, "card_index", out int cardIndex))
        {
            SimProtocolHelpers.SendError("malformed_action", "\"play\" requires a numeric \"card_index\"", runState, player);
            return;
        }
        int targetIndex = TryReadInt(root, "target_index", out int t) ? t : -1;
        int allyIndex = TryReadInt(root, "ally_index", out int a) ? a : -1;

        IReadOnlyList<CardModel> hand = PileType.Hand.GetPile(player).Cards;
        if (cardIndex < 0 || cardIndex >= hand.Count)
        {
            SimProtocolHelpers.SendError("invalid_index",
                $"card_index {cardIndex} out of range (hand size {hand.Count})", runState, player);
            return;
        }

        CardModel card = hand[cardIndex];
        if (!card.CanPlay(out UnplayableReason reason, out AbstractModel? _))
        {
            SimProtocolHelpers.SendError("unplayable_card", $"{card.Id.Entry}: {reason}", runState, player);
            return;
        }

        Creature? target = ResolveTarget(card, targetIndex, allyIndex);
        if (card.TargetType == TargetType.AnyEnemy && target == null)
        {
            SendMissingTarget(targetIndex, card.Id.Entry, runState, player);
            return;
        }
        if (card.TargetType == TargetType.AnyAlly && target == null)
        {
            // The game refuses an AnyAlly play without a living teammate.
            SimProtocolHelpers.SendError("invalid_target",
                $"{card.Id.Entry}: no living teammate at allies[{allyIndex}]", runState, player);
            return;
        }

        Logger.Log($"[SimCombat] Playing {card.Id.Entry} (target_index={targetIndex}, ally_index={allyIndex})");
        await PlayCardAndWaitAsync(player, card, target, ct);
    }

    /// <summary>The <c>potion</c> verb: use or discard the potion in a slot.</summary>
    private static async Task ExecutePotionAsync(JsonElement root, Player player, RunState? runState, CancellationToken ct)
    {
        if (!TryReadInt(root, "slot", out int slot))
        {
            SimProtocolHelpers.SendError("malformed_action", "\"potion\" requires a numeric \"slot\"", runState, player);
            return;
        }
        int targetIndex = TryReadInt(root, "target_index", out int t) ? t : -1;
        int allyIndex = TryReadInt(root, "ally_index", out int a) ? a : -1;
        bool discard = root.TryGetProperty("discard", out JsonElement discardEl) &&
                       discardEl.ValueKind == JsonValueKind.True;

        PotionModel? potion = SafeGetPotionAtSlot(player, slot);
        if (potion == null)
        {
            SimProtocolHelpers.SendError("empty_potion_slot", $"potion slot {slot} is empty", runState, player);
            return;
        }

        if (discard)
        {
            // PotionCmd.Discard has no in-use guard; the game UI disables
            // Discard for a queued potion, so the bridge refuses it too.
            if (potion.IsQueued)
            {
                SimProtocolHelpers.SendError("potion_busy",
                    $"potion {potion.Id.Entry} in slot {slot} is already queued for use", runState, player);
                return;
            }
            Logger.Log($"[SimCombat] Discarding potion in slot {slot}");
            await DiscardPotionAndWaitAsync(player, slot, ct);
            return;
        }

        Creature? target = ResolvePotionTarget(player, potion, targetIndex, allyIndex);
        if (allyIndex >= 0 && target == null)
        {
            SimProtocolHelpers.SendError("invalid_target",
                $"ally_index {allyIndex} is no living teammate {potion.Id.Entry} can be thrown at", runState, player);
            return;
        }
        if (potion.TargetType == TargetType.AnyEnemy && target == null)
        {
            SendMissingTarget(targetIndex, potion.Id.Entry, runState, player);
            return;
        }
        if (potion.TargetType == TargetType.AnyAlly && target == null)
        {
            SimProtocolHelpers.SendError("target_required",
                $"{potion.Id.Entry} needs a living teammate and there is none", runState, player);
            return;
        }

        Logger.Log($"[SimCombat] Using potion in slot {slot} (target_index={targetIndex}, ally_index={allyIndex})");
        await UsePotionAndWaitAsync(player, potion, slot, target, ct);
    }

    /// <summary>An explicit index that resolved to nothing is invalid_target; no index at all is target_required.</summary>
    private static void SendMissingTarget(int targetIndex, string what, RunState? runState, Player? player)
    {
        if (targetIndex >= 0)
        {
            SimProtocolHelpers.SendError("invalid_target",
                $"target_index {targetIndex} is not a hittable enemy for {what}", runState, player);
        }
        else
        {
            SimProtocolHelpers.SendError("target_required",
                $"{what} needs an enemy target and none is hittable", runState, player);
        }
    }

    /// <summary>Reads an int32 property that must be a JSON number.</summary>
    private static bool TryReadInt(JsonElement root, string name, out int value)
    {
        value = 0;
        return root.TryGetProperty(name, out JsonElement el) &&
               el.ValueKind == JsonValueKind.Number &&
               el.TryGetInt32(out value);
    }

    /// <summary>The potion in <paramref name="slot"/>, or null (empty, negative or unreadable slot).</summary>
    internal static PotionModel? SafeGetPotionAtSlot(Player player, int slot)
    {
        if (slot < 0)
            return null;
        try
        {
            return player.GetPotionAtSlotIndex(slot);
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>
    /// Target for a potion use. AnyEnemy follows the same addressing as
    /// <see cref="ResolveTarget"/>; self/any-player potions target the player;
    /// ally potions a teammate (<see cref="TeammateTarget"/>, null when none
    /// is alive); everything else auto-targets (null).
    /// <para>A non-negative <paramref name="allyIndex"/> throws an AnyPlayer
    /// potion at that teammate in <c>allies[]</c> instead (GH #1017; C#
    /// <c>PotionModel.CanThrowAtAlly</c>); null when it cannot be thrown or
    /// names no living teammate, which the caller refuses.</para>
    /// </summary>
    internal static Creature? ResolvePotionTarget(Player player, PotionModel? potion, int targetIndex, int allyIndex = -1)
    {
        if (potion == null)
            return null;
        if (allyIndex >= 0)
        {
            if (potion.TargetType != TargetType.AnyPlayer || !potion.CanThrowAtAlly())
                return null;
            return TeammateTarget(player, allyIndex);
        }

        switch (potion.TargetType)
        {
            case TargetType.AnyEnemy:
            {
                ICombatState? combat = player.Creature?.CombatState;
                if (combat == null)
                    return null;
                if (targetIndex < 0)
                    return combat.HittableEnemies.FirstOrDefault();
                if (targetIndex >= combat.Enemies.Count)
                    return null;
                Creature candidate = combat.Enemies[targetIndex];
                return candidate.IsHittable ? candidate : null;
            }
            case TargetType.AnyAlly:
                // Never the user (PotionModel.IsValidTarget).
                return TeammateTarget(player);
            case TargetType.Self:
            case TargetType.AnyPlayer:
                return player.Creature;
            default:
                return null;
        }
    }

    /// <summary>
    /// Settle-wait after requesting EndTurn -- the same Instant fast-mode
    /// async fix PlayCardAndWaitAsync got. PlayerCmd.EndTurn only REQUESTS
    /// the end of turn; under a busy action queue (multi-enemy fights, boss
    /// trigger storms) the player stays in the Play phase for many frames.
    /// Without this wait the turn loop's Until(IsPlayerInPlayPhase) gate is
    /// already true, the handler re-serializes the same Play-phase state, the
    /// agent (deterministically) answers end_turn again, and the duplicate
    /// EndTurn command lands in the NEXT turn's play phase -- ending it with
    /// zero cards played. Measured on the 2026-07-13 sweep: 209 duplicate
    /// re-sends across 9/12 runs; one death took 41 unblocked damage from a
    /// single eaten turn (seed 1005, floor 4).
    /// </summary>
    private static async Task WaitForTurnEndAsync(Player player, CancellationToken ct)
    {
        int waitMs = 0;
        while (waitMs < 3000)
        {
            if (!IsPlayerInPlayPhase(player) || !CombatManager.Instance.IsInProgress)
                return;
            await Task.Delay(16, ct);
            waitMs += 16;
        }
        Logger.Log("[SimCombat] WARNING: end turn still unprocessed after 3s; the agent will see a re-sent state");
    }

    private const int PlayStartWaitMs = 1000;
    private const int ResolutionSettleMs = 2000;

    /// <summary>
    /// Enqueues a card play and waits until it has really resolved.
    ///
    /// The start signal (energy paid / card left the hand) fires long before
    /// the card's effects, its move to discard/exhaust, cost-modifier cleanup
    /// and after-exhaust hooks (e.g. Feel No Pain's block) land; serializing at
    /// that point was the largest source of live mirror divergences. The
    /// action's State reaching Finished is strictly after all of that.
    /// Canceled is terminal but never completes the action's completion task,
    /// hence polling the state rather than awaiting it. GatheringPlayerChoice
    /// (or an open card selection) means the play is parked on a choice the
    /// agent answers on another path.
    /// </summary>
    private static async Task PlayCardAndWaitAsync(Player player, CardModel card, Creature? target, CancellationToken ct)
    {
        // Snapshot BEFORE enqueueing: under Instant fast mode the play can
        // resolve within the same frame.
        int energyBefore = player.PlayerCombatState?.Energy ?? -1;
        int handBefore = PileType.Hand.GetPile(player).Cards.Count;

        var action = new PlayCardAction(card, target);
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(action);

        // A free card that draws its own replacement changes neither signal,
        // hence the cap.
        await PollUntilAsync(() =>
            (player.PlayerCombatState?.Energy ?? -1) != energyBefore ||
            PileType.Hand.GetPile(player).Cards.Count != handBefore ||
            !IsPlayerInPlayPhase(player) ||
            !CombatManager.Instance.IsInProgress, PlayStartWaitMs, ct);

        if (!await WaitForResolutionAsync(action, player, ct))
        {
            Logger.Log($"[SimCombat] WARNING: {card.Id.Entry} still resolving after {ResolutionSettleMs}ms; " +
                       "the next state may be mid-resolution");
        }
    }

    /// <summary>Second settle phase shared by card plays and potion uses.</summary>
    private static Task<bool> WaitForResolutionAsync(GameAction action, Player player, CancellationToken ct) =>
        PollUntilAsync(() =>
            IsActionQuiescent(action) ||
            SimCardSelector.SelectionOpen ||
            !IsPlayerInPlayPhase(player) ||
            !CombatManager.Instance.IsInProgress, ResolutionSettleMs, ct);

    /// <summary>Polls <paramref name="condition"/> once per frame; false if it never held within <paramref name="maxMs"/>.</summary>
    private static async Task<bool> PollUntilAsync(Func<bool> condition, int maxMs, CancellationToken ct)
    {
        for (int waited = 0; ; waited += FrameSettleMs)
        {
            if (condition())
                return true;
            if (waited >= maxMs)
                return false;
            await Task.Delay(FrameSettleMs, ct);
        }
    }

    /// <summary>
    /// True once <paramref name="action"/> can no longer mutate game state on
    /// its own: it ran to completion, was cancelled, or suspended waiting on a
    /// player choice.
    ///
    /// GameAction.State is public (GameAction.cs:44) and is the engine's own
    /// end-of-action signal:
    ///   - Finished (GameAction.cs:139-146) is set in the `finally` of
    ///     GameAction.Execute only after _executionTask completed, i.e. after
    ///     the derived ExecuteAction's final `await` returned.
    ///   - Canceled (GameAction.cs:191) is terminal but never completes
    ///     _completionSource, so CompletionTask would never resolve there —
    ///     hence polling State instead of awaiting CompletionTask.
    ///   - GatheringPlayerChoice (GameAction.cs:183) means the action parked
    ///     itself on a player decision; the agent has to answer on another
    ///     channel before it can move again, so waiting here would deadlock
    ///     until the settle timeout.
    /// Every caller polls this under its own bounded timeout, so this can
    /// never turn into an unbounded wait.
    /// </summary>
    private static bool IsActionQuiescent(GameAction action)
    {
        return action.State == GameActionState.Finished
            || action.State == GameActionState.Canceled
            || action.State == GameActionState.GatheringPlayerChoice;
    }

    /// <summary>
    /// Enqueues a potion use and waits for it to resolve, with the same
    /// second settle phase as card plays (without it a draw potion was
    /// serialized with the drawn card still in the draw pile).
    /// </summary>
    private static async Task UsePotionAndWaitAsync(
        Player player, PotionModel potion, int slot, Creature? target, CancellationToken ct)
    {
        var action = new UsePotionAction(potion, target, CombatManager.Instance.IsInProgress);
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(action);

        // The slot empties at the very start of the use.
        await PollUntilAsync(() =>
            SafeGetPotionAtSlot(player, slot) == null ||
            !IsPlayerInPlayPhase(player) ||
            !CombatManager.Instance.IsInProgress, PlayStartWaitMs, ct);

        if (!await WaitForResolutionAsync(action, player, ct))
        {
            Logger.Log($"[SimCombat] WARNING: potion {potion.Id.Entry} still resolving after {ResolutionSettleMs}ms; " +
                       "the next state may be mid-resolution");
        }
    }

    /// <summary>
    /// Discards the potion in <paramref name="slot"/> the way the game's own
    /// Discard button does (NPotionPopup.OnDiscardButtonPressed, identical in
    /// v0.107.1 and v0.111.0): a <c>DiscardPotionGameAction</c> through the
    /// ActionQueueSynchronizer, so every co-op peer discards it too.
    ///
    /// It used to call <c>PotionCmd.Discard</c> directly, which changes only
    /// THIS machine's belt. In co-op the teammate's copy kept the potion, so
    /// the reward this discard made room for could not be taken there: its
    /// reward set never completed and both games hung on the next room's
    /// sync (2-instance run, Potion Courier, 2026-11-08).
    /// </summary>
    internal static async Task DiscardPotionAndWaitAsync(
        Player player, int slot, CancellationToken ct)
    {
        PotionModel? potion = SafeGetPotionAtSlot(player, slot);
        if (potion == null)
            return;

        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(
            new DiscardPotionGameAction(player, (uint)slot, CombatManager.Instance.IsInProgress));

        // A co-op client's discard round-trips through the host first.
        int waitMs = 0;
        while (waitMs < 3000)
        {
            if (SafeGetPotionAtSlot(player, slot) == null)
                break;
            await Task.Delay(16, ct);
            waitMs += 16;
        }
    }

    // ----------------------------------------------------------------
    // State serialization
    //
    // Phase 1a: delegates to SimStateExtractor for everything shared
    // (envelope, cards, piles, creatures/intents, potions). This handler
    // now only assembles the combat-specific shape around that.
    // ----------------------------------------------------------------

    /// <summary>
    /// Extra turns this player has taken this combat (PaelsEye, v0.111.0
    /// Ambergris), for the seat history's `extra_turns_taken_this_combat`
    /// (GH #812). TurnNumber and RoundNumber both start at 1 and step together
    /// at the side switch, except that an extra turn steps only the taking
    /// players' TurnNumber (decomp CombatManager.cs, the side switch; same in
    /// v0.107.1 and v0.111.0). The difference is therefore the count. A board
    /// rebuilt without it reports turn = round to the turn-number gates
    /// (BagOfPreparation, Lantern, the turn-1 Innate floor).
    /// </summary>
    internal static int ExtraTurnsTakenThisCombat(PlayerCombatState? pcs, CombatState? combatState)
    {
        if (pcs == null || combatState == null)
            return 0;
        return Math.Max(0, pcs.TurnNumber - combatState.RoundNumber);
    }

    /// <summary>
    /// Player card plays started this turn, bucketed by card type in the
    /// Rust engine's canonical order (Attack=0, Skill=1, Power=2, Curse=3,
    /// Status=4 — sts2-core cards.rs CardType::index). Mirrors the scan
    /// RingingPower.ShouldPlay does over CombatManager history.
    /// </summary>
    internal static int[] CountCardsPlayedThisTurn(CombatState? combatState, Creature? playerCreature)
    {
        var buckets = new int[5];
        if (combatState == null || playerCreature == null)
            return buckets;
        foreach (var entry in CombatManager.Instance.History.CardPlaysStarted)
        {
            if (!entry.HappenedThisTurn(combatState))
                continue;
            if (entry.CardPlay.Card.Owner.Creature != playerCreature)
                continue;
            int idx = entry.CardPlay.Card.Type switch
            {
                CardType.Attack => 0,
                CardType.Skill => 1,
                CardType.Power => 2,
                CardType.Curse => 3,
                CardType.Status => 4,
                _ => -1,
            };
            if (idx >= 0)
                buckets[idx]++;
        }
        return buckets;
    }

    /// <summary>
    /// ChainsOfBinding's once-per-turn gate, read from the POWER'S OWN FLAG.
    ///
    /// This field decides whether a second Bound card may be played
    /// (`ChainsOfBindingPower.ShouldPlay` returns `!boundCardPlayed`), and it
    /// is the single largest source of disagreement between the search and the
    /// live game: 22 of the 41 plays the game refused across the first 51
    /// ascension-climb runs were `BlockedByHook / CHAINS_OF_BINDING_POWER`,
    /// every one of them in a Queen fight. Each cost three refusals and then a
    /// thrown-away turn.
    ///
    /// It used to be RE-DERIVED here by walking `History.CardPlaysStarted` and
    /// re-testing each played card's affliction, on the reasoning that
    /// afflictions are only cleared at turn end so the scan reproduces the
    /// power exactly. Reproducing another object's private bookkeeping is a
    /// standing bet that every one of its edge cases was noticed --
    /// `BeforeCardPlayed`'s dupe/owner filters, `BeforeSideTurnEnd`'s reset
    /// point, and whatever the history collection does or does not record --
    /// and it is a bet with no upside, because the flag itself is reachable.
    /// Read the authority; do not model it. (Same lesson as every other
    /// SpireMentor-synthesized wire field.)
    ///
    /// The history scan survives as the fallback for a build whose internals
    /// moved, so a renamed field degrades to the old behavior instead of
    /// silently reporting `false` -- which is the answer that causes the
    /// refusals.
    /// </summary>
    private static bool BoundCardPlayedThisTurn(CombatState? combatState, Creature? playerCreature)
    {
        if (combatState == null || playerCreature == null)
            return false;
        bool? own = ReadChainsBoundFlag(playerCreature);
        if (own.HasValue)
            return own.Value;
        foreach (var entry in CombatManager.Instance.History.CardPlaysStarted)
        {
            if (!entry.HappenedThisTurn(combatState))
                continue;
            if (entry.CardPlay.Card.Owner.Creature != playerCreature)
                continue;
            if (entry.CardPlay.Card.IsDupe)
                continue;
            if (entry.CardPlay.Card.Affliction is Bound)
                return true;
        }
        return false;
    }

    /// <summary>Logged once: a reflection failure repeats every poll.</summary>
    private static bool _chainsFlagUnreadable;

    /// <summary>
    /// `ChainsOfBindingPower`'s private `boundCardPlayed`, or null when it
    /// cannot be read (caller falls back to the history scan).
    ///
    /// `PowerModel._internalData` holds an instance of the power's private
    /// nested `Data` class (`PowerModel.cs:45`, v0.107.1); the power itself
    /// reaches it through `GetInternalData&lt;T&gt;()`, which is `protected`.
    /// Both hops are by name so a build that renames either one falls back
    /// rather than throwing.
    ///
    /// No power on the creature means no card is Bound and nothing is gated,
    /// which is `false` -- not "unreadable".
    /// </summary>
    private static bool? ReadChainsBoundFlag(Creature playerCreature)
    {
        try
        {
            PowerModel? chains = playerCreature.Powers
                .FirstOrDefault(p => p is ChainsOfBindingPower);
            if (chains == null)
                return false;
            object? data = typeof(PowerModel)
                .GetField("_internalData", BindingFlags.NonPublic | BindingFlags.Instance)
                ?.GetValue(chains);
            object? flag = data?.GetType()
                .GetField("boundCardPlayed",
                          BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                ?.GetValue(data);
            if (flag is bool value)
                return value;
        }
        catch (Exception ex)
        {
            if (!_chainsFlagUnreadable)
            {
                _chainsFlagUnreadable = true;
                Logger.Log(
                    "[SimCombat] Could not read ChainsOfBindingPower.boundCardPlayed " +
                    $"({ex.Message}); falling back to the history scan.");
            }
            return null;
        }
        if (!_chainsFlagUnreadable)
        {
            _chainsFlagUnreadable = true;
            Logger.Log(
                "[SimCombat] ChainsOfBindingPower.boundCardPlayed not found on this build; " +
                "falling back to the history scan.");
        }
        return null;
    }

    private const string SerializationFailedJson = "{\"type\":\"combat_action\",\"error\":\"serialization_failed\"}";

    /// <summary>The combat_action message for the agent, and its state_id (-1 on failure).</summary>
    private (string json, long stateId) SerializeCombatState(Player player)
    {
        try
        {
            Dictionary<string, object>? state = BuildCombatState(player);
            if (state == null)
                return (SerializationFailedJson, -1);
            return (JsonSerializer.Serialize(state), (long)state["state_id"]);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCombat] Could not serialize combat state: {ex}");
            return (SerializationFailedJson, -1);
        }
    }

    /// <summary>
    /// The one builder of the combat board (<c>combat_action</c>), envelope
    /// applied. The advisor and the board embedded in card_select reuse it,
    /// so there is a single shape. Null (logged) on failure.
    /// </summary>
    internal static Dictionary<string, object>? BuildCombatState(Player player)
    {
        try
        {
            CombatState? combatState = CombatManager.Instance.DebugOnlyGetState();
            Creature? creature = player.Creature;
            PlayerCombatState? pcs = player.PlayerCombatState;
            RunState? runState = RunManager.Instance.DebugOnlyGetState();
            IReadOnlyList<Creature> foes = combatState?.Enemies ?? new List<Creature>();

            var playerInfo = new Dictionary<string, object>
            {
                ["hp"] = creature?.CurrentHp ?? 0,
                ["max_hp"] = creature?.MaxHp ?? 0,
                ["block"] = creature?.Block ?? 0,
                ["energy"] = pcs?.Energy ?? 0,
                ["max_energy"] = pcs?.MaxEnergy ?? 3,
                ["powers"] = creature?.Powers.Select(SimStateExtractor.SerializePower).ToList()
                             ?? new List<Dictionary<string, object?>>(),
                ["cards_played_this_turn"] = CountCardsPlayedThisTurn(combatState, creature),
                ["bound_card_played_this_turn"] = BoundCardPlayedThisTurn(combatState, creature),
                ["hp_lost_this_turn"] = SimCombatLog.LostHpThisTurn(creature),
                // History counts a rebuilt board cannot redo (GH #989): Tear
                // Asunder's hits, and the hits a teammate's Gang Up counts.
                ["unblocked_hits_this_combat"] = SimCombatLog.UnblockedHitsThisCombat(creature),
                ["powered_hits_this_turn"] = SimCombatLog.PoweredHitsThisTurn(creature, foes),
                // Our exhausts: a Midnight entering combat counts everyone's.
                ["cards_exhausted_this_combat"] = SimCombatLog.CardsExhaustedThisCombat(creature),
                // This turn's exhausts (Evil Eye) and card block gains
                // (Unmovable), which a rebuilt board cannot recount (GH #989).
                ["cards_exhausted_this_turn"] = SimCombatLog.CardsExhaustedThisTurn(creature),
                ["card_block_gains_this_turn"] = SimCombatLog.CardBlockGainsThisTurn(creature),
                // TurnNumber - RoundNumber: the turn-number gates read the
                // turn, and a rebuilt board otherwise sets it to the round.
                ["extra_turns_taken_this_combat"] = ExtraTurnsTakenThisCombat(pcs, combatState),
                // Co-op: who we are on the combat log's card-play rows (`owner`).
                ["net_id"] = player.NetId.ToString(System.Globalization.CultureInfo.InvariantCulture),
                // Our place in the run's player order: who sets up, draws and
                // flushes first each turn (they share the run's shuffle stream).
                ["seat_index"] = runState?.Players.ToList().IndexOf(player) ?? 0,
            };
            try
            {
                if (pcs != null && pcs.Stars >= 0 && player.Character?.Id.Entry == "REGENT")
                    playerInfo["stars"] = pcs.Stars;
            }
            catch (Exception ex)
            {
                Logger.Log($"[SimCombat] Could not read stars: {ex.Message}");
            }

            List<Dictionary<string, object?>> hand = combatState != null && pcs != null
                ? PileType.Hand.GetPile(player).Cards
                    .Select(c => SimStateExtractor.SerializeCard(c, includeLegality: true))
                    .ToList()
                : new List<Dictionary<string, object?>>();

            List<Dictionary<string, object?>> enemies = foes
                .Select(e => SimStateExtractor.SerializeCreature(e))
                .ToList();

            List<Dictionary<string, object?>> orbs = SimStateExtractor.SerializeOrbs(player, out int orbCapacity);

            List<Dictionary<string, object?>> potions = player.PotionSlots
                .Select((potion, slot) => SimStateExtractor.SerializePotionSlot(potion, slot))
                .ToList();

            var state = new Dictionary<string, object>
            {
                ["type"] = "combat_action",
                ["player"] = playerInfo,
                ["hand"] = hand,
                ["draw_pile"] = SimStateExtractor.SerializePile(pcs?.DrawPile),
                ["discard_pile"] = SimStateExtractor.SerializePile(pcs?.DiscardPile),
                ["exhaust_pile"] = SimStateExtractor.SerializePile(pcs?.ExhaustPile),
                // A card whose OnPlay awaits a selection sits in the play pile
                // until the pick resolves; a board serialized during that
                // screen must still show it.
                ["play_pile"] = SimStateExtractor.SerializePile(pcs?.PlayPile),
                ["enemies"] = enemies,
                ["orbs"] = orbs,
                ["orb_capacity"] = orbCapacity,
                ["pets"] = SimStateExtractor.SerializePets(player),
                ["potions"] = potions,
                ["available_actions"] = GetAvailableActions(potions),
                ["draw_pile_count"] = pcs?.DrawPile?.Cards.Count ?? 0,
                ["discard_pile_count"] = pcs?.DiscardPile?.Cards.Count ?? 0,
                ["exhaust_pile_count"] = pcs?.ExhaustPile?.Cards.Count ?? 0,
                ["round"] = combatState?.RoundNumber ?? 0,
                ["room_type"] = runState?.CurrentRoom?.RoomType.ToString() ?? "Combat",
                ["allies"] = SerializeAllies(player, runState, foes),
            };

            SimStateExtractor.ApplyEnvelope(state, runState, player);
            return state;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCombat] Could not build combat state: {ex}");
            return null;
        }
    }

    /// <summary>
    /// Co-op (PROTOCOL.md "allies"): every other player in the run, as this
    /// peer simulates them. Every peer runs every player's draws and plays,
    /// so a teammate's hand is known here even though the game's own UI only
    /// shows its size. Empty in single-player. One unreadable teammate is
    /// logged and left out rather than failing the board.
    /// </summary>
    private static List<Dictionary<string, object?>> SerializeAllies(
        Player me, RunState? runState, IReadOnlyList<Creature> foes)
    {
        var allies = new List<Dictionary<string, object?>>();
        if (runState == null)
            return allies;
        foreach (Player other in runState.Players)
        {
            if (ReferenceEquals(other, me))
                continue;
            try
            {
                Creature? c = other.Creature;
                PlayerCombatState? pcs = other.PlayerCombatState;
                List<Dictionary<string, object?>> orbs =
                    SimStateExtractor.SerializeOrbs(other, out int orbCapacity);
                allies.Add(new Dictionary<string, object?>
                {
                    ["character"] = other.Character?.Id.Entry ?? "",
                    ["hp"] = c?.CurrentHp ?? 0,
                    ["max_hp"] = c?.MaxHp ?? 0,
                    ["block"] = c?.Block ?? 0,
                    ["is_dead"] = c?.IsDead ?? true,
                    ["energy"] = pcs?.Energy ?? 0,
                    ["max_energy"] = pcs?.MaxEnergy ?? 0,
                    ["powers"] = c?.Powers.Select(SimStateExtractor.SerializePower).ToList()
                                 ?? new List<Dictionary<string, object?>>(),
                    ["hand"] = pcs != null
                        ? PileType.Hand.GetPile(other).Cards
                            .Select(card => SimStateExtractor.SerializeCard(card))
                            .ToList()
                        : new List<Dictionary<string, object?>>(),
                    ["ended_turn"] = SafeEndedTurn(other),
                    // The rest of the seat, so a recorded fight can be replayed
                    // as a whole co-op fight: every peer runs every player's draws.
                    ["net_id"] = other.NetId.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    ["draw_pile"] = SimStateExtractor.SerializePile(pcs?.DrawPile),
                    ["discard_pile"] = SimStateExtractor.SerializePile(pcs?.DiscardPile),
                    ["exhaust_pile"] = SimStateExtractor.SerializePile(pcs?.ExhaustPile),
                    ["play_pile"] = SimStateExtractor.SerializePile(pcs?.PlayPile),
                    ["relics"] = other.Relics.Select(SimStateExtractor.SerializeRelic).ToList(),
                    ["stars"] = pcs?.Stars ?? 0,
                    ["hp_lost_this_turn"] = SimCombatLog.LostHpThisTurn(c),
                    // Their plays this turn, as ours: their Ripple Basin and
                    // Juggling count Attacks (GH #989).
                    ["cards_played_this_turn"] = CountCardsPlayedThisTurn(
                        CombatManager.Instance.DebugOnlyGetState(), c),
                    ["unblocked_hits_this_combat"] = SimCombatLog.UnblockedHitsThisCombat(c),
                    ["powered_hits_this_turn"] = SimCombatLog.PoweredHitsThisTurn(c, foes),
                    ["cards_exhausted_this_combat"] = SimCombatLog.CardsExhaustedThisCombat(c),
                    ["cards_exhausted_this_turn"] = SimCombatLog.CardsExhaustedThisTurn(c),
                    ["card_block_gains_this_turn"] = SimCombatLog.CardBlockGainsThisTurn(c),
                    ["extra_turns_taken_this_combat"] = ExtraTurnsTakenThisCombat(
                        pcs, CombatManager.Instance.DebugOnlyGetState()),
                    // Their belt, slot by slot as ours is: a replay plays a
                    // teammate's logged potion from it.
                    ["potions"] = other.PotionSlots
                        .Select((potion, slot) => SimStateExtractor.SerializePotionSlot(potion, slot))
                        .ToList(),
                    // Their orb belt, as ours: Chaos channels orbs for any
                    // character, and a Frost orb blocks at their turn end.
                    ["orbs"] = orbs,
                    ["orb_capacity"] = orbCapacity,
                    // Their pets, as ours (GH #1017): a Necrobinder's Osty
                    // soaks the hits on them and deals their Osty attacks.
                    ["pets"] = SimStateExtractor.SerializePets(other),
                });
            }
            catch (Exception ex)
            {
                // Never drop the seat: `allies[]` indices name teammates (an
                // AnyAlly play's `ally_index`, TeammateTarget), so a missing
                // entry would shift every later one. Send what reads safely;
                // without a hand the engine sees HP and block only.
                Logger.Log($"[SimCombat] Could not serialize a teammate: {ex.Message}");
                allies.Add(MinimalAlly(other));
            }
        }
        return allies;
    }

    /// <summary>A teammate's <c>allies[]</c> entry when the full one throws:
    /// identity, HP and block, each read on its own.</summary>
    private static Dictionary<string, object?> MinimalAlly(Player other)
    {
        static T Read<T>(Func<T> f, T fallback)
        {
            try
            {
                return f();
            }
            catch (Exception)
            {
                return fallback;
            }
        }
        return new Dictionary<string, object?>
        {
            ["character"] = Read(() => other.Character?.Id.Entry ?? "", ""),
            ["hp"] = Read(() => other.Creature?.CurrentHp ?? 0, 0),
            ["max_hp"] = Read(() => other.Creature?.MaxHp ?? 0, 0),
            ["block"] = Read(() => other.Creature?.Block ?? 0, 0),
            ["is_dead"] = Read(() => other.Creature?.IsDead ?? true, true),
            ["net_id"] = Read<string?>(
                () => other.NetId.ToString(System.Globalization.CultureInfo.InvariantCulture), null),
        };
    }

    private static bool SafeEndedTurn(Player player)
    {
        try
        {
            return CombatManager.Instance.IsInProgress
                && CombatManager.Instance.IsPlayerReadyToEndTurn(player);
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>
    /// <see cref="BuildCombatState"/> for EMBEDDING inside another decision
    /// message, or null when no fight is in progress.
    ///
    /// Drops <c>state_id</c>: the agent gate keys a response to the top-level
    /// decision it answers, and a second id nested inside it is a trap for
    /// anyone who reaches for the wrong one. A failure is logged and reported
    /// as "no board" rather than thrown — the consumer decides how serious
    /// that is, and the combat mirror treats a missing board on an in-combat
    /// selection as a divergence rather than guessing.
    /// </summary>
    internal static Dictionary<string, object>? BuildEmbeddedCombatState(Player? player)
    {
        try
        {
            if (player == null || !CombatManager.Instance.IsInProgress)
                return null;
            Dictionary<string, object>? state = BuildCombatState(player);
            state?.Remove("state_id");
            return state;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCombat] Could not embed the combat board: {ex.Message}");
            return null;
        }
    }

    /// <summary>PLAY and END_TURN always; POTION when any slot reports can_use=true.</summary>
    private static List<string> GetAvailableActions(IEnumerable<Dictionary<string, object?>> potions)
    {
        var actions = new List<string> { "PLAY", "END_TURN" };
        if (potions.Any(p => p.TryGetValue("can_use", out object? canUse) && canUse is true))
            actions.Add("POTION");
        return actions;
    }
}
