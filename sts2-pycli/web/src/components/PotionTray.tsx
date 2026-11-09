import type { Potion, Score } from "../api";
import { useAdviceMarks } from "../advice/useAdviceMarks";
import { potionKey } from "../advice/marks";

/**
 * The potion belt, as the game's top bar carries it: a recessed pod of
 * sockets (one per slot), the potion sprite sitting in each filled socket.
 * Name, resolved description, the win-prob hint and the Use/Discard actions
 * live in a popover that opens on hover, tap or keyboard focus — the controls
 * are ALWAYS in the DOM (visibility is pure CSS), so replay tests and
 * assistive tech see the same tree as live play. Without handlers (replay,
 * off-combat screens) the popover is display-only.
 *
 * It lives in the HUD band (GameHud), where the game keeps it; the advice's
 * mark rings the socket the AI would drink from, in the same gold as the
 * cards. Slot indices are the RAW `potions` index (a hole stays a hole), which
 * is what `use_potion` / `discard_potion` take.
 */
export default function PotionTray({
  potions,
  slots,
  scores,
  onUse,
  onDiscard,
}: {
  potions: (Potion | null | undefined)[];
  slots?: number;
  scores?: Score[];
  onUse?: (slot: number) => void;
  onDiscard?: (slot: number) => void;
}) {
  // Render at least the owned potions; pad to `slots` empty sockets if known.
  const total = Math.max(potions.length, slots ?? potions.length);
  // The advice's mark on a slot (advice/marks.ts); the engine's number only
  // with "engine numbers" on.
  const { marks, showNumbers } = useAdviceMarks();
  const tiles = Array.from({ length: total }, (_, i) => potions[i]);
  if (total === 0) return null;

  return (
    <span className="gs-hud-potions" data-testid="hud-potions">
      {tiles.map((pt, slot) => {
        if (!pt) {
          return (
            <span key={slot} className="gs-hud-socket gs-potion-empty" role="img"
              aria-label="empty potion slot" />
          );
        }
        // The recommender scores a potion as stable_idx -100 - slot.
        const sc = showNumbers ? scores?.find((s) => s.stable_idx === -100 - slot) : undefined;
        const mark = marks.get(potionKey(slot)) ?? null;
        return (
          <span key={slot} className={`gs-potion${mark ? ` gs-mark-${mark.tier}` : ""}`}
            data-testid={mark ? "potion-mark" : undefined}>
            <span className="gs-hud-socket gs-potion-filled" tabIndex={0}
              role="img" aria-label={pt.name}>
              {pt.image
                ? <img className="gs-potion-art" src={pt.image} alt="" draggable={false} />
                : <span className="gs-potion-glyph" aria-hidden="true">◉</span>}
              {sc && <span className="gs-potion-win">{sc.win_prob.toFixed(2)}</span>}
              {mark?.order != null && (
                <span className={`gs-badge gs-step-tag gs-step-${mark.tier}`} aria-label={`step ${mark.order}`}>
                  {mark.order}
                </span>
              )}
            </span>
            <span className="gs-potion-pop" role="tooltip">
              <span className="gs-potion-popin">
                <span className="gs-potion-popname">{pt.name}</span>
                {pt.description && <span className="gs-potion-popdesc">{pt.description}</span>}
                {(onUse || onDiscard) && (
                  <span className="gs-potion-actions">
                    {onUse && (
                      <button type="button" className="gs-potion-btn" onClick={() => onUse(slot)}>
                        use
                      </button>
                    )}
                    {onDiscard && (
                      <button type="button" className="gs-potion-btn gs-potion-btn-dim"
                        aria-label={`discard ${pt.name}`} onClick={() => onDiscard(slot)}>
                        discard
                      </button>
                    )}
                  </span>
                )}
              </span>
            </span>
          </span>
        );
      })}
    </span>
  );
}
