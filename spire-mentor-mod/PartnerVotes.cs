// PartnerVotes.cs -- the once-only keys an AI partner's votes are sent under.
//
// SimAiPartnerBrain sends each vote once per key (`Once`), so the key decides
// how often a partner can vote. It must change exactly when the game opens a
// new round of that vote.
//
// Game-free on purpose (no Godot, no sts2.dll) so tests/unit can link it.

namespace SpireMentor;

internal static class PartnerVotes
{
    /// <summary>
    /// The key for a partner's vote to move on (ActChangeSynchronizer).
    ///
    /// <para>The act index alone is not enough. The last act holds TWO
    /// rounds: after the boss (RunManager.EnterNextAct enters The Architect,
    /// because there is no next act), and The Architect's Proceed
    /// (TheArchitect.WinRun, then EnterNextAct wins the run). Both happen in
    /// the same act, so a key on the act index let the partners vote only the
    /// first time, and every co-op run with AI partners hung at the Architect,
    /// "waiting for players" (public issue #47). ActChangeSynchronizer.
    /// MoveToNextAct increments ActFloor after each round, so the act floor
    /// tells the two rounds apart.</para>
    ///
    /// <para>Only v0.107.1 has the second round. On v0.111.0 the Architect's
    /// Proceed calls RunManager.WinRun directly, with no vote.</para>
    /// </summary>
    public static string ActChange(int actIndex, int actFloor) => $"act:{actIndex}:{actFloor}";
}
