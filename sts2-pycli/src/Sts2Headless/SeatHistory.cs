using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Combat.History.Entries;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.ValueProps;

namespace Sts2Headless;

/// <summary>
/// The combat-history counts a rebuilt board cannot recount (GH #989, #1712),
/// the same scans and wire keys as the mod (`spire-mentor-mod/SimCombatLog.cs`).
/// Each one is something the player can see or count: Spite glows when you
/// lost HP this turn, Tear Asunder prints its hits, Unmovable's doubling and
/// Evil Eye's bonus follow from plays made in front of you.
/// </summary>
internal static class SeatHistory
{
    public static void Add(Dictionary<string, object?> summary, Player player, PlayerCombatState? pcs)
    {
        var creature = player.Creature;
        var state = CombatManager.Instance?.DebugOnlyGetState();
        if (creature == null || state == null) return;
        var history = CombatManager.Instance!.History;
        var damage = history.Entries.OfType<DamageReceivedEntry>().ToList();
        var exhausts = history.Entries.OfType<CardExhaustedEntry>().ToList();

        summary["hp_lost_this_turn"] = damage.Any(e =>
            e.HappenedThisTurn(state) && e.Receiver == creature && e.Result.UnblockedDamage > 0);
        summary["unblocked_hits_this_combat"] = damage.Count(e =>
            e.Receiver == creature && e.Result.UnblockedDamage > 0);
        summary["cards_exhausted_this_combat"] = exhausts.Count(e => e.Actor == creature);
        summary["cards_exhausted_this_turn"] = exhausts.Count(e => e.HappenedThisTurn(state) && e.Actor == creature);
        // The cards the player discarded this turn (`CardDiscardedEntry`,
        // Memento Mori's multiplier). A discard before the first screen
        // (Whispering Earring's turn-1 auto-plays) is otherwise lost.
        summary["discards_this_turn"] = history.Entries.OfType<CardDiscardedEntry>()
            .Count(e => e.HappenedThisTurn(state) && e.Actor == creature);
        // Murder's multiplier (`Murder.cs`: every `CardDrawnEntry` whose Actor
        // is the owner, this combat). The card's `calculateddamage` stat is
        // the unscaled base, so without this count a rebuilt Murder hits for 1.
        summary["cards_drawn_this_combat"] = history.Entries.OfType<CardDrawnEntry>().Count(e => e.Actor == creature);
        // Supermassive's multiplier (`Supermassive.cs`: every
        // `CardGeneratedEntry` whose Creator is the owner). A card generated
        // before the first screen (Radiant Pearl's Luminesce) is otherwise
        // missing from a rebuilt fight.
        summary["cards_generated_this_combat"] = history.Entries.OfType<CardGeneratedEntry>()
            .Count(e => e.Creator != null && e.Creator.Creature == creature);

        var finished = history.CardPlaysFinished
            .Where(f => f.CardPlay != null).Select(f => f.CardPlay).ToHashSet();
        summary["card_block_gains_this_turn"] = history.Entries.OfType<BlockGainedEntry>().Count(e =>
            e.HappenedThisTurn(state) && e.CardPlay != null && finished.Contains(e.CardPlay)
            && e.CardPlay.Card?.Owner?.Creature == creature && e.Props.IsCardOrMonsterMove());

        var enemies = state.Enemies;
        var hits = damage.Where(e => e.Dealer == creature && e.Result.Props.IsPoweredAttack()
            && e.HappenedThisTurn(state)).ToList();
        var rows = new List<Dictionary<string, object?>>();
        for (var i = 0; i < enemies.Count; i++)
        {
            var n = hits.Count(e => e.Receiver == enemies[i]);
            if (n > 0) rows.Add(new Dictionary<string, object?> { ["enemy"] = i, ["hits"] = n });
        }
        summary["powered_hits_this_turn"] = rows;

        if (pcs != null)
            summary["extra_turns_taken_this_combat"] = Math.Max(0, pcs.TurnNumber - state.RoundNumber);
    }
}
