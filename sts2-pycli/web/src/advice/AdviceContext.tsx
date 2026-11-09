// Supplies the advice marks (what they are and the hooks that read them live
// in useAdviceMarks.ts), and the step badge a marked thing wears.

import { useMemo, type ReactNode } from "react";
import type { Mark } from "./marks";
import { AdviceMarksCtx, type AdviceMarksValue } from "./useAdviceMarks";

export function AdviceMarksProvider({ value, children }: { value: AdviceMarksValue; children: ReactNode }) {
  return <AdviceMarksCtx.Provider value={value}>{children}</AdviceMarksCtx.Provider>;
}

/** Test helper: marks without App. */
export function MarksFixture({ marks, showNumbers = false, children }: {
  marks: Mark[]; showNumbers?: boolean; children: ReactNode;
}) {
  const value = useMemo(() => ({ marks: new Map(marks.map((m) => [m.key, m])), showNumbers }),
    [marks, showNumbers]);
  return <AdviceMarksCtx.Provider value={value}>{children}</AdviceMarksCtx.Provider>;
}

/** The step badge drawn on a marked thing (only numbered marks get one). */
export function MarkBadge({ mark, className = "" }: { mark: Mark | null; className?: string }) {
  if (!mark || mark.order == null) return null;
  return (
    <span className={`wb-mark-badge wb-mark-badge--${mark.tier} ${className}`} aria-label={`step ${mark.order}`}>
      {mark.order}
    </span>
  );
}
