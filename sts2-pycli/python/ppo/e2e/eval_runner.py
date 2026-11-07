"""The learner's eval runner (ops contract item 8): one claimed eval request.

The learner claims a request from ``<lane>/evals/queue/`` and runs this module
on it at nice 10::

    python -m ppo.e2e.eval_runner REQUEST --out OUT [--evals DIR] [--pin DIR]
        [--device cpu] [--policy ppo.e2e.net:make_student] [EVALUATE_RUN_ARGS ...]

It does three things:

1. Writes the request's ``variant`` (``raw``, ``ema0996`` or ``ema0999``) of
   the request's ``state`` as a WP4b net checkpoint, ``OUT/<variant>.pt``
   (`update.export_variant`). A rerun of the same request reuses the file, so
   its sha256 stays the same and ``ppo.e2e.evaluate`` keeps the shards it
   already finished. The state's sha256 must match the request's ``sha256``.
2. Plays the checkpoint's greedy student (`net.make_student`) on the request's
   seeds with ``ppo.e2e.evaluate run``. The arm, under ``<evals>/DEV/``, is
   the request's ``id``, the state's first 8 sha256 characters and a hash of
   the stack (``git`` HEAD and dirty flag, pin, device, policy): a rerun on
   the same stack resumes the arm's finished shards, and a rerun after a
   deploy starts a new arm instead of failing ``evaluate``'s config check.
   An ops panel (``DEV-1000``, ``DEV-2000``) is a slice of the runner panel
   ``DEV``. The runner never plays a sealed panel (TEST).
3. Reads the arm with ``ppo.e2e.evaluate``'s DEV report and writes
   ``OUT/result.json``: ``wins`` and ``runs`` (the finished runs, which
   exclude error and timeout rows) plus the report. The learner adds that
   file's fields to ``evals/done/<id>.json``; the driver reads a result only
   when ``runs`` equals the request's ``n``.

``--pin`` defaults to ``E2E_PIN``, then ``STS2_WHEEL_PIN`` (the watchdog sets
the latter for the learner); ``--evals`` defaults to ``E2E_EVALS``, then
``~/runs/e2e/evals``. Arguments the runner doesn't know go to
``ppo.e2e.evaluate run`` unchanged (for example ``--allow-unpinned`` in a test).
Exit codes: 0 when the report exists and covers the request's ``n`` runs, 1
when it covers fewer, 2 when the request is refused or ``evaluate`` fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import ops_contract as OC
from . import panels as P

DEFAULT_POLICY = "ppo.e2e.net:make_student"
RESULT = "result.json"
#: Ops panels (`ops_contract.EVAL_PANELS`) -> the runner panel they slice.
OPS_PANELS = dict.fromkeys(OC.EVAL_PANELS, "DEV")


class RunnerError(RuntimeError):
    pass


def runner_panel(name: str) -> P.Panel:
    """The runner panel of an ops panel (DEV-1000 -> DEV) or of a runner panel name."""
    try:
        panel = P.runner_panel(OPS_PANELS.get(name, name))
    except P.PanelError as e:
        raise RunnerError(str(e)) from None
    if panel.sealed:
        raise RunnerError(f"{panel.name} is sealed; the learner never plays it")
    return panel


def arm_name(req: Mapping[str, Any], state_sha256: str, stack: str = "") -> str:
    """The request's id, the state's sha256 prefix and the stack id, as a safe
    directory name."""
    rid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(req.get("id") or "eval")).strip("._") or "eval"
    return f"{rid}_{state_sha256[:8]}" + (f"_{stack}" if stack else "")


def stack_id(pin: str | None, device: str, policy: str) -> str:
    """8 hex characters over what `evaluate` writes into an arm's config besides
    the checkpoint: the checkout's HEAD and dirty flag, the pin, device, policy."""
    from .evaluate import git_head

    sha, dirty = git_head()
    text = f"{sha}|{int(dirty)}|{pin or '-'}|{device}|{policy}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def default_pin() -> str | None:
    return os.environ.get("E2E_PIN") or os.environ.get("STS2_WHEEL_PIN") or None


def default_evals() -> str:
    return os.environ.get("E2E_EVALS") or str(Path.home() / "runs" / "e2e" / "evals")


def read_report(evals: Path, panel: P.Panel, arm: str) -> dict[str, Any]:
    """``ppo.e2e.evaluate report --json`` of one arm, in-process."""
    from . import evaluate as E
    from . import score_test

    try:
        a = score_test.load_arm(evals, panel, arm, sealed_rules=False)
    except score_test.ReadStopped as e:
        raise RunnerError(f"integrity: {e}") from None
    return E.report([a.rows[s] for s in sorted(a.rows)])


def evaluate_argv(
    arm: str,
    panel: P.Panel,
    n: int,
    seed_start: int,
    *,
    ckpt: Path,
    procs: int,
    evals: Path,
    pin: str | None,
    device: str,
    policy: str,
    extra: Sequence[str] = (),
) -> list[str]:
    argv = [sys.executable, "-m", "ppo.e2e.evaluate", "run", arm, panel.name]
    argv += [str(n), str(seed_start), "--policy", policy, "--ckpt", str(ckpt)]
    argv += ["--procs", str(procs), "--device", device, "--evals", str(evals)]
    if pin:
        argv += ["--pin", pin]
    return argv + list(extra)


def run_request(
    request: Path,
    out: Path,
    *,
    evals: Path,
    pin: str | None,
    device: str = "cpu",
    policy: str = DEFAULT_POLICY,
    extra: Sequence[str] = (),
    run: Callable[..., Any] = subprocess.run,
    report: Callable[[Path, P.Panel, str], dict[str, Any]] = read_report,
) -> dict[str, Any]:
    """Run one request; returns (and writes to ``out/result.json``) the result."""
    from . import ckpt
    from .update import export_variant

    req = OC.read_json(request)
    if not isinstance(req, dict):
        raise RunnerError(f"{request}: not a JSON object")
    try:
        state = Path(str(req["state"]))
        n, seed_start = int(req["n"]), int(req["seed_start"])
    except (KeyError, TypeError, ValueError) as e:
        raise RunnerError(f"{request}: needs state, n and seed_start ({e})") from None
    variant = str(req.get("variant") or "raw")
    panel = runner_panel(str(req.get("panel") or ""))
    try:
        P.check_slice(panel, seed_start, n)
    except P.PanelError as e:
        raise RunnerError(str(e)) from None
    if not state.is_file():
        raise RunnerError(f"state {state}: no such file")
    sha = ckpt.sha256_file(state)
    if req.get("sha256") and req["sha256"] != sha:
        raise RunnerError(f"state {state}: sha256 {sha} is not the request's {req['sha256']}")
    out.mkdir(parents=True, exist_ok=True)
    student = out / f"{variant}.pt"
    if not student.is_file():
        export_variant(str(state), variant, str(student))
    arm = arm_name(req, sha, stack_id(pin, device, policy))
    procs = max(1, min(int(req.get("max_procs") or OC.EVAL_MAX_PROCS), OC.EVAL_MAX_PROCS))
    argv = evaluate_argv(
        arm,
        panel,
        n,
        seed_start,
        ckpt=student,
        procs=procs,
        evals=evals,
        pin=pin,
        device=device,
        policy=policy,
        extra=extra,
    )
    from .evaluate import CLEAN_MARKER

    env = {k: v for k, v in os.environ.items() if k != CLEAN_MARKER}
    print("[eval_runner] " + " ".join(argv), flush=True)
    rc = run(argv, env=env).returncode
    if rc != 0:
        raise RunnerError(f"ppo.e2e.evaluate run exited {rc}")
    rep = report(evals, panel, arm)
    res = {
        "wins": int(rep["wins"]),
        "runs": int(rep["scored"]),
        "complete": int(rep["scored"]) == n,
        "arm": arm,
        "arm_dir": str(evals / panel.name / arm),
        "student": str(student),
        "student_sha256": ckpt.sha256_file(student),
        "state_sha256": sha,
        "report": rep,
    }
    OC.atomic_write_text(out / RESULT, json.dumps(res, indent=1, sort_keys=True) + "\n")
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ppo.e2e.eval_runner", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("request")
    ap.add_argument("--out", required=True)
    ap.add_argument("--evals", default=None)
    ap.add_argument("--pin", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    args, extra = ap.parse_known_args(argv)
    evals = Path(args.evals or default_evals()).expanduser().resolve()
    try:
        res = run_request(
            Path(args.request),
            Path(args.out),
            evals=evals,
            pin=args.pin or default_pin(),
            device=args.device,
            policy=args.policy,
            extra=extra,
        )
    except RunnerError as e:
        print(f"[eval_runner] refused: {e}", file=sys.stderr, flush=True)
        return 2
    print(f"[eval_runner] {res['arm']}: {res['wins']}/{res['runs']}", flush=True)
    return 0 if res["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
