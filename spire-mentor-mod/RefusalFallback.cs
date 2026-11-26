// RefusalFallback.cs -- which move autoplay tries after the game refuses the
// recommended one.
//
// The advisor is deterministic for a given board, so re-asking after a refusal
// returns the same refused move; the old escape hatch then ended the turn after
// three of them. On 2026-11-23 a targeting bug in the advisor (#1875) turned
// that into lost turns in autobattle: "recommended target no longer hittable"
// three times, end turn, on every turn left in the fight, with energy and a
// full hand unplayed. Every advice reply already ranks the other moves, so the
// next one in its order is a far better fallback than ending the turn.
//
// The refused moves hold for one board: they are keyed by the combat-history
// mark (SimAdvisor.CombatHistoryMark), which a refused move leaves unchanged and
// any accepted one moves, so a hand index refused now never blocks the same
// index on a later board.
//
// Game-free on purpose (no Godot, no sts2.dll) so tests/unit can link it.

using System.Collections.Generic;
using System.Linq;

namespace SpireMentor;

internal sealed class RefusalFallback
{
    private readonly HashSet<string> _refused = new();
    private int _mark = int.MinValue;

    /// <summary>A combat move's identity on one board.</summary>
    public static string Key(string action, int? cardIndex, int? targetIndex, int? allyIndex, int? slot) =>
        $"{action}|{cardIndex ?? -1}|{targetIndex ?? -1}|{allyIndex ?? -1}|{slot ?? -1}";

    public static string Key(MarkOption o) => Key(o.Action, o.CardIndex, o.TargetIndex, o.AllyIndex, o.Slot);

    /// <summary>How many moves were refused on the board at <paramref name="mark"/>.</summary>
    public int RefusedOn(int mark) => mark == _mark ? _refused.Count : 0;

    /// <summary>The game refused <paramref name="key"/> on the board at <paramref name="mark"/>.</summary>
    public void Refuse(int mark, string key)
    {
        if (mark != _mark)
        {
            _refused.Clear();
            _mark = mark;
        }
        _refused.Add(key);
    }

    /// <summary>Whether <paramref name="key"/> was refused on the board at <paramref name="mark"/>.</summary>
    public bool IsRefused(int mark, string key) => mark == _mark && _refused.Contains(key);

    /// <summary>
    /// The advice's best combat move not refused on the board at
    /// <paramref name="mark"/>, in the advice's own preference order: the
    /// engine's pick order (`pick_rank`) first, then its share (visit share or
    /// the net's probability), then the order it listed them in. Only a move
    /// autoplay can execute counts: a card play with its hand index, a potion
    /// with its slot, or ending the turn. Null when none is left.
    /// </summary>
    public MarkOption? Next(int mark, IReadOnlyList<MarkOption> options) =>
        options
            .Select((o, i) => (o, i))
            .Where(p => Executable(p.o) && !IsRefused(mark, Key(p.o)))
            .OrderBy(p => p.o.PickRank ?? int.MaxValue)
            .ThenByDescending(p => p.o.Share ?? double.NegativeInfinity)
            .ThenBy(p => p.i)
            .Select(p => p.o)
            .FirstOrDefault();

    private static bool Executable(MarkOption o) => o.Action switch
    {
        "play" => o.CardIndex is >= 0,
        "potion" => o.Slot is >= 0,
        "end_turn" => true,
        _ => false,
    };
}
