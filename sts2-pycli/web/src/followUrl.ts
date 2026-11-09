// The follow-link URL the mod opens (SimWorkbenchLink.cs):
// `?follow=1[&run=<name>][&step=<idx>]`. `step` is a replay step index (the
// position of a `state` line in the run log) that a coach record carries, so
// "open in workbench" lands on that decision instead of the newest step.

export interface FollowParams {
  follow: boolean;
  run: string | null;
  step: number | null;
}

export function parseFollowParams(search: string): FollowParams {
  const params = new URLSearchParams(search);
  const raw = params.get("step");
  const step = raw !== null && /^\d+$/.test(raw) ? Number(raw) : null;
  return { follow: params.get("follow") === "1", run: params.get("run"), step };
}

/** `?open=<path>` (the seed checker's "open in the replay viewer"): the
 *  stored run log to open, or null. */
export function parseOpenParam(search: string): string | null {
  const v = new URLSearchParams(search).get("open");
  return v && v.trim() ? v : null;
}

/** The step to open a run of `n` steps at: the newest without a `step`, else
 *  `step` clamped into the run (a link to a step not yet mirrored opens at the
 *  tail rather than nowhere). */
export function seekIndex(step: number | null, n: number): number {
  if (n <= 0) return 0;
  if (step === null) return n - 1;
  return Math.min(Math.max(step, 0), n - 1);
}

/** The run-log name the advisor's control plane gives a path (`GET /log`'s
 *  `name`, realgame_workbench_log.run_log_name): the basename with anything
 *  outside `[A-Za-z0-9._-]` replaced by `_`. */
export function runLogName(path: string): string {
  const base = path.split(/[\\/]/).pop() ?? "";
  return base.replace(/[^A-Za-z0-9._-]/g, "_");
}

/** Which run to open for a follow link. A coach link names its run by the
 *  control plane's bare `name`; when that is the run the advisor is writing,
 *  open it by the live path instead — the dev workbench loads runs by path,
 *  and a bare name resolves to nothing there ("no run loaded"). */
export function resolveRunPath(run: string | null, live: string | null): string | null {
  if (run && live && runLogName(live) === run) return live;
  return run ?? live;
}
