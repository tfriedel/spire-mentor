import type { CombatLogRow } from "../api";

/**
 * The one place a combat-log display row becomes a coloured sentence.
 *
 * Shared by both history logs (LiveLog / EventLog), so a hit reads
 * identically everywhere: actors coloured by side, amounts and sources in
 * gold.
 */

/** Colour for a creature name: the player's side reads cool, the enemy's hot. */
function sideColor(side: string | null): string {
  return side === "enemy" ? "var(--hp)" : "var(--block)";
}

/** The coloured spans for one row, falling back to its plain `text`. */
export function LogLine({ row }: { row: CombatLogRow }) {
  const actor = row.actor ? (
    <span style={{ color: sideColor(row.actor_side), fontWeight: 600 }}>{row.actor}</span>
  ) : null;
  const target = row.target ? (
    <span style={{ color: sideColor(row.target_side), fontWeight: 600 }}>{row.target}</span>
  ) : null;
  const amount = (
    <span style={{ color: "var(--gold)", fontWeight: 700 }}>{row.amount}</span>
  );
  const source = row.source ? (
    <>
      {" using "}
      <span style={{ color: "var(--gold)" }}>{row.source}</span>
    </>
  ) : null;
  const noun = <span style={{ color: "var(--gold)" }}>{row.noun}</span>;

  switch (row.kind) {
    case "damage":
      return actor ? (
        <>{actor} dealt {amount} damage to {target}{source}.</>
      ) : (
        <>{target} took {amount} damage.</>
      );
    case "block":
      return <>{actor} gained {amount} {noun}.</>;
    case "power":
      // "applied N X to Y" vs "gained N X" — the server decided which (a
      // self-applied power has actor === target and still reads "applied").
      return row.verb === "applied" ? (
        <>{actor} applied {amount} {noun} to {target}.</>
      ) : (
        <>{target} gained {amount} {noun}.</>
      );
    case "monster_move":
      return <>{actor} used {noun}.</>;
    case "potion":
      return <>{actor} drank {noun}.</>;
    case "orb_channeled":
      return <>{actor} channeled {noun}.</>;
    case "orb_evoked":
      return <>{actor} evoked {noun}.</>;
    // A card the engine played on its own (Stampede, Havoc, Mayhem, …). The
    // step label only ever names the card the player chose, so this line is
    // where the auto-play shows (GH #1155). `source` is what pulled it.
    case "card_auto_played":
      return (
        <>
          {actor} auto-played {noun}
          {row.source ? <>{" via "}<span style={{ color: "var(--gold)" }}>{row.source}</span></> : null}.
        </>
      );
    // A card a play moved: Brand's exhaust pick, Survivor's discard, an added
    // Shiv (GH #1015). The step label names only the card played.
    case "card_exhausted":
    case "card_discarded":
    case "card_generated":
      return row.actor && row.verb && row.noun ? (
        <>{actor} {row.verb} {noun}.</>
      ) : (
        <>{row.text}</>
      );
    // Every other kind (including anything a newer engine adds) renders the
    // server's own sentence rather than being dropped.
    default:
      return <>{row.text}</>;
  }
}
