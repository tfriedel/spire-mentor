// SimCombatLog.cs -- the running game's own combat log, on the wire.
//
// The game records every hit, block, power, move and card movement in
// `CombatManager.Instance.History` (MegaCrit.Sts2.Core.Combat.History), which
// is exactly what the workbench's combat-log panel shows for a SIMULATED run.
// This maps the real game's entries into the SAME row shape the Rust engine
// emits (`sts2_core::event_log` -> `RunManager.combat_log()`), so a live game
// followed in the workbench renders through the identical Python/React path
// with no second renderer to keep in sync.
//
// Polled, not subscribed: `SimAdvisor` already reads `History.Entries` for its
// settle heuristic, the list only grows within a fight, and a poll cannot leak
// a handler when a combat ends mid-teardown. `Drain()` returns everything
// recorded since the previous call and resets itself when the entry count
// SHRINKS -- the only way that happens is a new combat's fresh History.
//
// Every mapper is defensive: this is display data and a null-ref here would
// take the advice request down with it, so anything unexpected degrades to the
// entry's own `Description` string.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Combat.History;
using MegaCrit.Sts2.Core.Combat.History.Entries;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.ValueProps;

namespace SpireMentor;

internal static class SimCombatLog
{
    /// <summary>How many History entries have already been sent.</summary>
    private static int _sent;

    /// <summary>
    /// Rows drained into a request that never went out (the advisor was
    /// gone or the write failed): the next <see cref="Drain"/> sends them
    /// first, so a replay does not lose a teammate's plays to a dropped
    /// connection (GH #989). Main thread only, like Drain.
    /// </summary>
    private static readonly List<Dictionary<string, object?>> _undelivered = new();

    /// <summary>
    /// The move the acting monster is mid-way through, so its damage rows can
    /// carry "using Butt Move" the way the game's log does. C# hands a
    /// `CardSource` to `DamageReceivedEntry` for card damage but nothing for a
    /// monster move, so the move has to be remembered as the entries stream
    /// past -- the same nesting the Rust engine's log derives.
    /// </summary>
    private static string? _openMoveId;
    private static string? _openMoveMonster;

    /// <summary>`Rd 3 (Player turn): ...` -- the header C#
    /// `CombatHistoryEntry.HumanReadableString` prefixes every line with. The
    /// round and side are PRIVATE on the entry, so this is the only way to read
    /// the values the entry itself was stamped with (rather than the values the
    /// board happens to hold when we drain).</summary>
    private static readonly Regex HeaderRe = new(
        @"^Rd (\d+) \((\w+) turn\)", RegexOptions.Compiled);

    /// <summary>
    /// `player_choice` rows not yet sent, each with the History entry count
    /// at the moment of the choice: Drain puts a row before the first entry
    /// recorded after it. Choices are not History entries; they arrive by
    /// event (<see cref="RecordChoice"/>), possibly off the main thread.
    /// </summary>
    private static readonly List<(int At, Dictionary<string, object?> Row)> _choices = new();

    private static readonly object _choicesLock = new();

    /// <summary>
    /// The cards each player was last offered on a choose-a-card screen (a
    /// Skill Potion, Discovery), by net id, until their pick arrives: the
    /// game syncs only the picked index, and the offer is random, so a
    /// replay that diverges needs the offer to tell why (GH #989). Under
    /// <see cref="_choicesLock"/>.
    /// </summary>
    private static readonly Dictionary<string, List<Dictionary<string, object?>>> _offers = new();

    /// <summary>Forget everything (a new run / a restored snapshot).</summary>
    internal static void Reset()
    {
        _sent = 0;
        // Another fight's rows: sent with this one's they would read as its.
        _undelivered.Clear();
        _openMoveId = null;
        _openMoveMonster = null;
        lock (_choicesLock)
        {
            // A choice made before the new combat's History began (an
            // opening Toolbox pick) still belongs to it.
            for (int i = 0; i < _choices.Count; i++)
            {
                _choices[i] = (0, _choices[i].Row);
            }
        }
    }

    /// <summary>
    /// The cards <paramref name="owner"/> is offered on a choose-a-card
    /// screen, put on their next `Index` pick as `offered` (C#
    /// `CardSelectCmd.FromChooseACardScreen` runs on every peer, the
    /// chooser's and each teammate's). Never throws.
    /// </summary>
    internal static void RecordOffer(string? owner, List<Dictionary<string, object?>> cards)
    {
        lock (_choicesLock)
        {
            _offers[owner ?? ""] = cards;
        }
    }

    /// <summary>
    /// A player's choice mid-combat, as the game syncs it between peers (C#
    /// `PlayerChoiceSynchronizer.PlayerChoiceReceived`, raised for ours and
    /// for every teammate's): who chose, the per-player choice id, the
    /// result's type (`CombatCard`, `CanonicalCard`, `Index`, ...) and what
    /// was picked, cards as `{id, upgraded}` in the order chosen. Lets a
    /// recorded co-op fight replay every seat's picks. Never throws.
    /// </summary>
    internal static void RecordChoice(
        string? owner, uint choiceId, string type,
        List<Dictionary<string, object?>> cards, List<int>? indexes, string? player)
    {
        try
        {
            if (CombatManager.Instance?.IsInProgress != true)
            {
                // A rest-site Smith or an event pick: not the fight's.
                return;
            }
            CombatState? state = CombatManager.Instance.DebugOnlyGetState();
            int at = CombatManager.Instance?.History?.Entries?.Count() ?? 0;
            var row = new Dictionary<string, object?>
            {
                ["round"] = state?.RoundNumber,
                ["side"] = state == null ? null : state.CurrentSide == CombatSide.Player ? "player" : "enemy",
                ["text"] = cards.Count > 0
                    ? $"{owner} chose {string.Join(", ", cards.Select(c => c["id"]))}"
                    : $"{owner} chose {type} {string.Join(", ", indexes ?? new List<int>())}{player}",
                ["source"] = null,
                ["kind"] = "player_choice",
                ["owner"] = owner,
                ["choice_id"] = choiceId,
                ["choice_type"] = type,
                ["cards"] = cards,
                ["indexes"] = indexes,
                ["player"] = player,
            };
            lock (_choicesLock)
            {
                if (type == "Index" && _offers.Remove(owner ?? "", out var offered))
                {
                    row["offered"] = offered;
                }
                _choices.Add((at, row));
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCombatLog] RecordChoice failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Whether <paramref name="creature"/> lost HP this turn: C#
    /// `Spite.LostHpThisTurn`, a History scan for unblocked damage received
    /// this turn. On the board as `hp_lost_this_turn`, which a rebuilt state
    /// cannot recount (a Spite after Inferno's self-damage hit once). False on
    /// any error.
    /// </summary>
    internal static bool LostHpThisTurn(Creature? creature)
    {
        try
        {
            CombatState? state = CombatManager.Instance?.DebugOnlyGetState();
            if (creature == null || state == null)
            {
                return false;
            }
            return CombatManager.Instance!.History.Entries.OfType<DamageReceivedEntry>().Any(e =>
                e.HappenedThisTurn(state) && e.Receiver == creature && e.Result.UnblockedDamage > 0);
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>
    /// The hits that took <paramref name="creature"/>'s HP this combat: C#
    /// `TearAsunder`'s hit count, a History scan for damage received with
    /// unblocked damage. On the board as `unblocked_hits_this_combat` (GH
    /// #989). 0 on any error.
    /// </summary>
    internal static int UnblockedHitsThisCombat(Creature? creature)
    {
        try
        {
            if (creature == null || CombatManager.Instance?.DebugOnlyGetState() == null)
            {
                return 0;
            }
            return CombatManager.Instance.History.Entries.OfType<DamageReceivedEntry>().Count(e =>
                e.Receiver == creature && e.Result.UnblockedDamage > 0);
        }
        catch (Exception)
        {
            return 0;
        }
    }

    /// <summary>
    /// The cards <paramref name="creature"/>'s player exhausted this combat
    /// (`CardExhaustedEntry`, whose actor is the card's owner). A Midnight
    /// entering combat starts cheaper by every player's. On the board as
    /// `cards_exhausted_this_combat`. 0 on any error.
    /// </summary>
    internal static int CardsExhaustedThisCombat(Creature? creature)
    {
        try
        {
            if (creature == null || CombatManager.Instance?.DebugOnlyGetState() == null)
            {
                return 0;
            }
            return CombatManager.Instance.History.Entries.OfType<CardExhaustedEntry>()
                .Count(e => e.Actor == creature);
        }
        catch (Exception)
        {
            return 0;
        }
    }

    /// <summary>
    /// The cards <paramref name="creature"/>'s player exhausted this turn: C#
    /// `EvilEye.WasCardExhaustedThisTurn` and `ForgottenRitual`. On the board as
    /// `cards_exhausted_this_turn`, which a rebuilt state cannot recount (a
    /// Toasty Mittens exhaust at turn start, GH #989). 0 on any error.
    /// </summary>
    internal static int CardsExhaustedThisTurn(Creature? creature)
    {
        try
        {
            CombatState? state = CombatManager.Instance?.DebugOnlyGetState();
            if (creature == null || state == null)
            {
                return 0;
            }
            return CombatManager.Instance!.History.Entries.OfType<CardExhaustedEntry>()
                .Count(e => e.HappenedThisTurn(state) && e.Actor == creature);
        }
        catch (Exception)
        {
            return 0;
        }
    }

    /// <summary>
    /// The block gains from <paramref name="creature"/>'s FINISHED card plays
    /// this turn: what v0.111.0 `UnmovablePower.ModifyBlockMultiplicative`
    /// counts before it doubles a card's block (`BlockGainedEntry`s this turn
    /// with `CardPlay.Player.Creature == Owner`, `IsCardOrMonsterMove`, from
    /// another play). `CardPlay.Player` is the card's owner
    /// (`CardModel.cs:1918`), read here as `CardPlay.Card.Owner`, which both
    /// builds have. A play still resolving (a board sent while its pick is
    /// open) is left out: C# never counts a play's own gains, and the engine
    /// cannot tell them from an earlier play's once rebuilt. v0.107.1 counted
    /// by receiver instead (`e.Actor == target`, any move block included);
    /// this sends the v0.111.0 count. On the board as
    /// `card_block_gains_this_turn` (GH #989: rebuilt without it, Unmovable
    /// doubled a second Defend). 0 on any error.
    /// </summary>
    internal static int CardBlockGainsThisTurn(Creature? creature)
    {
        try
        {
            CombatState? state = CombatManager.Instance?.DebugOnlyGetState();
            if (creature == null || state == null)
            {
                return 0;
            }
            var history = CombatManager.Instance!.History;
            var finished = history.CardPlaysFinished
                .Where(f => f.CardPlay != null).Select(f => f.CardPlay).ToHashSet();
            return history.Entries.OfType<BlockGainedEntry>().Count(e =>
                e.HappenedThisTurn(state) && e.CardPlay != null && finished.Contains(e.CardPlay)
                && e.CardPlay.Card?.Owner?.Creature == creature && e.Props.IsCardOrMonsterMove());
        }
        catch (Exception)
        {
            return 0;
        }
    }

    /// <summary>
    /// The cards <paramref name="creature"/>'s player discarded this turn
    /// (`CardDiscardedEntry`, Memento Mori's multiplier). A discard before the
    /// first screen (Whispering Earring's turn-1 auto-plays) is otherwise lost.
    /// On the board as `discards_this_turn`, the host's key
    /// (`Sts2Headless/SeatHistory.cs`, GH #1895). 0 on any error.
    /// </summary>
    internal static int DiscardsThisTurn(Creature? creature)
    {
        try
        {
            CombatState? state = CombatManager.Instance?.DebugOnlyGetState();
            if (creature == null || state == null)
            {
                return 0;
            }
            return CombatManager.Instance!.History.Entries.OfType<CardDiscardedEntry>()
                .Count(e => e.HappenedThisTurn(state) && e.Actor == creature);
        }
        catch (Exception)
        {
            return 0;
        }
    }

    /// <summary>
    /// The cards <paramref name="creature"/>'s player drew this combat: C#
    /// `Murder`'s multiplier (every `CardDrawnEntry` whose Actor is the
    /// owner). The card's `calculateddamage` stat is the unscaled base, so a
    /// rebuilt Murder without this count hits for 1. On the board as
    /// `cards_drawn_this_combat`, the host's key (GH #1868). 0 on any error.
    /// </summary>
    internal static int CardsDrawnThisCombat(Creature? creature)
    {
        try
        {
            if (creature == null || CombatManager.Instance?.DebugOnlyGetState() == null)
            {
                return 0;
            }
            return CombatManager.Instance.History.Entries.OfType<CardDrawnEntry>()
                .Count(e => e.Actor == creature);
        }
        catch (Exception)
        {
            return 0;
        }
    }

    /// <summary>
    /// The cards generated for <paramref name="creature"/>'s player this
    /// combat: C# `Supermassive`'s multiplier (every `CardGeneratedEntry`
    /// whose Creator is the owner). A card generated before the first screen
    /// (Radiant Pearl's Luminesce) is otherwise missing from a rebuilt fight.
    /// On the board as `cards_generated_this_combat`, the host's key. 0 on any
    /// error.
    /// </summary>
    internal static int CardsGeneratedThisCombat(Creature? creature)
    {
        try
        {
            if (creature == null || CombatManager.Instance?.DebugOnlyGetState() == null)
            {
                return 0;
            }
            return CombatManager.Instance.History.Entries.OfType<CardGeneratedEntry>()
                .Count(e => e.Creator != null && e.Creator.Creature == creature);
        }
        catch (Exception)
        {
            return 0;
        }
    }

    /// <summary>
    /// The powered hits <paramref name="dealer"/> landed on each enemy this
    /// turn, as `[{enemy, hits}]` by the enemy's place in
    /// <paramref name="enemies"/>: what a teammate's C# `GangUp` counts. On the
    /// board as `powered_hits_this_turn` (GH #989). Empty on any error.
    /// </summary>
    internal static List<Dictionary<string, object?>> PoweredHitsThisTurn(
        Creature? dealer, IReadOnlyList<Creature> enemies)
    {
        var rows = new List<Dictionary<string, object?>>();
        try
        {
            CombatState? state = CombatManager.Instance?.DebugOnlyGetState();
            if (dealer == null || state == null)
            {
                return rows;
            }
            List<DamageReceivedEntry> hits = CombatManager.Instance!.History.Entries
                .OfType<DamageReceivedEntry>()
                .Where(e => e.Dealer == dealer && e.Result.Props.IsPoweredAttack()
                    && e.HappenedThisTurn(state))
                .ToList();
            for (int i = 0; i < enemies.Count; i++)
            {
                int n = hits.Count(e => e.Receiver == enemies[i]);
                if (n > 0)
                {
                    rows.Add(new Dictionary<string, object?> { ["enemy"] = i, ["hits"] = n });
                }
            }
        }
        catch (Exception)
        {
            rows.Clear();
        }
        return rows;
    }

    /// <summary>A card as a `player_choice` row names it.</summary>
    internal static Dictionary<string, object?> ChoiceCard(string? id, bool upgraded) =>
        new()
        {
            ["id"] = id,
            ["upgraded"] = upgraded,
        };

    /// <summary>The buffered choice rows made before History entry
    /// <paramref name="before"/>, removed from the buffer, oldest first.</summary>
    private static List<Dictionary<string, object?>> TakeChoices(int before)
    {
        var rows = new List<Dictionary<string, object?>>();
        lock (_choicesLock)
        {
            for (int i = 0; i < _choices.Count;)
            {
                if (_choices[i].At <= before)
                {
                    rows.Add(_choices[i].Row);
                    _choices.RemoveAt(i);
                }
                else
                {
                    i++;
                }
            }
        }
        return rows;
    }

    /// <summary>
    /// Rows recorded since the last call, oldest first. Empty outside combat,
    /// on the first call of a fresh combat's first decision, or on any error.
    /// Never throws.
    /// </summary>
    internal static List<Dictionary<string, object?>> Drain()
    {
        var rows = new List<Dictionary<string, object?>>();
        try
        {
            List<CombatHistoryEntry>? entries =
                CombatManager.Instance?.History?.Entries?.ToList();
            if (entries == null)
            {
                return rows;
            }
            if (entries.Count < _sent)
            {
                // A shorter history than last time can only be a new combat.
                Reset();
            }
            rows.AddRange(_undelivered);
            _undelivered.Clear();
            for (int i = _sent; i < entries.Count; i++)
            {
                rows.AddRange(TakeChoices(i));
                Dictionary<string, object?>? row = Map(entries[i]);
                if (row != null)
                {
                    rows.Add(row);
                }
            }
            _sent = entries.Count;
            rows.AddRange(TakeChoices(entries.Count));
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCombatLog] Drain failed: {ex.Message}");
        }
        return rows;
    }

    /// <summary>
    /// Give back the rows of a request that was not sent: the next
    /// <see cref="Drain"/> returns them ahead of its own. Never throws.
    /// </summary>
    internal static void Requeue(List<Dictionary<string, object?>>? rows)
    {
        if (rows != null && rows.Count > 0)
        {
            _undelivered.InsertRange(0, rows);
        }
    }

    /// <summary>One history entry as a wire row, or null if it maps to nothing.</summary>
    private static Dictionary<string, object?>? Map(CombatHistoryEntry entry)
    {
        Dictionary<string, object?> row = Header(entry);
        switch (entry)
        {
            case DamageReceivedEntry e:
                row["kind"] = "damage";
                row["dealer"] = Who(e.Dealer);
                row["receiver"] = Who(e.Receiver);
                row["unblocked"] = e.Result.UnblockedDamage + e.Result.OverkillDamage;
                row["blocked"] = e.Result.BlockedDamage;
                row["source"] = e.CardSource != null
                    ? CardSource(e.CardSource.Id.Entry, e.CardSource.IsUpgraded)
                    : OpenMoveSource();
                return row;

            case BlockGainedEntry e:
                row["kind"] = "block";
                row["receiver"] = Who(e.Actor);
                row["amount"] = e.Amount;
                row["source"] = e.CardPlay?.Card != null
                    ? CardSource(e.CardPlay.Card.Id.Entry, e.CardPlay.Card.IsUpgraded)
                    : OpenMoveSource();
                return row;

            case PowerReceivedEntry e:
                row["kind"] = "power";
                row["applier"] = Who(e.Applier);
                row["receiver"] = Who(e.Actor);
                row["power"] = e.Power?.Id.Entry;
                row["amount"] = (int)e.Amount;
                row["source"] = OpenMoveSource();
                return row;

            case MonsterPerformedMoveEntry e:
                row["kind"] = "monster_move";
                row["monster"] = Who(e.Actor);
                row["move_id"] = e.Move?.Id;
                // Open the move so the hits it deals can name it.
                _openMoveId = e.Move?.Id;
                _openMoveMonster = e.Monster?.Id.Entry;
                return row;

            case CardPlayStartedEntry e:
                row["kind"] = "card_play_started";
                row["card"] = e.CardPlay?.Card?.Id.Entry;
                row["upgraded"] = e.CardPlay?.Card?.IsUpgraded ?? false;
                row["auto_played"] = e.CardPlay?.IsAutoPlay ?? false;
                // Co-op: whose card and at what, so a recorded fight can be
                // replayed seat by seat. `play_index` > 0 is a repeat of the
                // same play (Echo Form, Burst), not a new one. The player is
                // the entry's Actor, the card's owner when the play started
                // (CardPlayStartedEntry passes cardPlay.Card.Owner.Creature):
                // the card's own Owner can change before these rows are read.
                // The Ball is handed to a teammate after its play
                // (CardPileCmd.GiveToAnotherPlayer), and read through Owner
                // its row named the teammate (GH #1017).
                row["owner"] = NetIdOf(() => e.Actor?.Player?.NetId ?? e.CardPlay?.Card?.Owner?.NetId);
                row["target"] = Who(e.CardPlay?.Target);
                row["target_index"] = EnemyIndex(e.CardPlay?.Target);
                row["play_index"] = e.CardPlay?.PlayIndex ?? 0;
                return row;

            case CardPlayFinishedEntry e:
                row["kind"] = "card_play_finished";
                row["card"] = e.CardPlay?.Card?.Id.Entry;
                row["upgraded"] = e.CardPlay?.Card?.IsUpgraded ?? false;
                return row;

            case CardGeneratedEntry e:
                row["kind"] = "card_generated";
                row["card"] = e.Card?.Id.Entry;
                row["upgraded"] = e.Card?.IsUpgraded ?? false;
                row["by_player"] = e.Creator != null;
                return row;

            case CardDrawnEntry e:
                row["kind"] = "card_drawn";
                row["card"] = e.Card?.Id.Entry;
                row["upgraded"] = e.Card?.IsUpgraded ?? false;
                row["from_hand_draw"] = e.FromHandDraw;
                return row;

            case CardDiscardedEntry e:
                row["kind"] = "card_discarded";
                row["card"] = e.Card?.Id.Entry;
                row["upgraded"] = e.Card?.IsUpgraded ?? false;
                return row;

            case CardExhaustedEntry e:
                row["kind"] = "card_exhausted";
                row["card"] = e.Card?.Id.Entry;
                row["upgraded"] = e.Card?.IsUpgraded ?? false;
                // Whose card: the entry's actor is its owner (GH #989).
                row["owner"] = NetIdOf(() => e.Actor?.Player?.NetId);
                return row;

            case OrbChanneledEntry e:
                row["kind"] = "orb_channeled";
                row["orb"] = e.Orb?.Id.Entry;
                return row;

            case PotionUsedEntry e:
                row["kind"] = "potion";
                row["potion"] = e.Potion?.Id.Entry;
                row["owner"] = NetIdOf(() => e.Potion?.Owner?.NetId);
                // As on card plays: an index can go stale once an enemy dies.
                row["target"] = Who(e.Target);
                row["target_index"] = EnemyIndex(e.Target);
                return row;

            case EnergySpentEntry e:
                row["kind"] = "energy_spent";
                row["amount"] = e.Amount;
                return row;

            default:
                // An entry kind the workbench has no renderer for still reaches
                // the panel: `text` (below) is the game's own sentence, and the
                // Python side falls back to it for any unknown `kind`.
                row["kind"] = "other";
                return row;
        }
    }

    /// <summary>The fields every row carries: round, side and the game's own text.</summary>
    private static Dictionary<string, object?> Header(CombatHistoryEntry entry)
    {
        var row = new Dictionary<string, object?>
        {
            ["round"] = null,
            ["side"] = null,
            ["text"] = DescriptionIsSafe(entry) ? Safe(() => entry.Description) : null,
            ["source"] = null,
        };
        string? human = DescriptionIsSafe(entry) ? Safe(() => entry.HumanReadableString) : null;
        if (human != null)
        {
            Match m = HeaderRe.Match(human);
            if (m.Success)
            {
                if (int.TryParse(m.Groups[1].Value, NumberStyles.Integer,
                        CultureInfo.InvariantCulture, out int round))
                {
                    row["round"] = round;
                }
                // C# `CombatSide` is Player / Monster; the wire says enemy.
                row["side"] = m.Groups[2].Value == "Player" ? "player" : "enemy";
            }
        }
        return row;
    }

    /// <summary>A creature as `{kind, id, uid}`, matching the engine's shape.</summary>
    private static Dictionary<string, object?>? Who(Creature? creature)
    {
        if (creature == null)
        {
            return null;
        }
        return Safe(() =>
        {
            var who = new Dictionary<string, object?>
            {
                ["kind"] = creature.IsPlayer ? "player" : "monster",
                ["id"] = creature.ModelId.Entry,
                ["uid"] = null,
                // Matches `combat_id` on the board's creatures (co-op replay).
                ["combat_id"] = creature.CombatId,
            };
            // A player's `net_id` (the board's), so a replay knows which
            // teammate an AnyAlly card went to (GH #1017).
            if (creature.IsPlayer)
                who["net_id"] = NetIdOf(() => creature.Player?.NetId);
            return who;
        });
    }

    /// <summary>A player's network id, as a string (a `ulong` does not
    /// survive every JSON reader). Matches `net_id` on the combat board.</summary>
    private static string? NetIdOf(Func<ulong?> netId)
    {
        try
        {
            return netId()?.ToString(CultureInfo.InvariantCulture);
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>The position of <paramref name="creature"/> among the combat's
    /// enemies, or null for a player, a pet or no target.</summary>
    private static int? EnemyIndex(Creature? creature)
    {
        if (creature == null)
        {
            return null;
        }
        try
        {
            int i = CombatManager.Instance?.DebugOnlyGetState()?.Enemies.ToList().IndexOf(creature) ?? -1;
            return i >= 0 ? i : null;
        }
        catch (Exception)
        {
            return null;
        }
    }

    private static Dictionary<string, object?> CardSource(string id, bool upgraded) =>
        new()
        {
            ["kind"] = "card",
            ["id"] = id,
            ["upgraded"] = upgraded,
        };

    /// <summary>The monster move currently mid-resolution, if any.</summary>
    private static Dictionary<string, object?>? OpenMoveSource()
    {
        if (_openMoveId == null)
        {
            return null;
        }
        return new Dictionary<string, object?>
        {
            ["kind"] = "move",
            ["monster"] = _openMoveMonster,
            ["move_id"] = _openMoveId,
        };
    }

    /// <summary>Run <paramref name="f"/>, swallowing anything it throws.</summary>
    // Entry types whose Description reads Actor.Player.Character with no null
    // check. By name: ArrowShotEntry exists on v0.113 only, and one DLL binds
    // against every build.
    private static readonly HashSet<string> PlayerActorDescriptions = new()
    {
        "CardGeneratedEntry", "CardExhaustedEntry", "CardAfflictedEntry", "SummonedEntry",
        "EnergySpentEntry", "OrbChanneledEntry", "CardDiscardedEntry", "CardDrawnEntry",
        "StarsModifiedEntry", "ArrowShotEntry", "PotionUsedEntry", "CardPlayStartedEntry",
        "CardPlayFinishedEntry",
    };

    /// <summary>
    /// False when the game's own <c>Description</c> getter (and
    /// <c>HumanReadableString</c>, which embeds it) would dereference null for
    /// this entry. The getters name a target as <c>Target.Monster.Id</c>, and a
    /// player creature has no Monster: every potion drunk on yourself, and a
    /// card played at a teammate. A try/catch is not enough. On Linux the
    /// modded game runs without the runtime's SIGSEGV handler (the Sentry
    /// shutdown removes it), so that null dereference kills the process
    /// instead of throwing (GH #1425).
    /// </summary>
    internal static bool DescriptionIsSafe(CombatHistoryEntry entry)
    {
        if (PlayerActorDescriptions.Contains(entry.GetType().Name) && entry.Actor?.Player?.Character == null)
        {
            return false;
        }
        Creature? target = entry switch
        {
            PotionUsedEntry potion => potion.Target,
            CardPlayStartedEntry started => started.CardPlay?.Target,
            CardPlayFinishedEntry finished => finished.CardPlay?.Target,
            _ => null,
        };
        return target == null || target.Monster != null;
    }

    private static T? Safe<T>(Func<T> f) where T : class
    {
        try
        {
            return f();
        }
        catch (Exception)
        {
            return null;
        }
    }
}
