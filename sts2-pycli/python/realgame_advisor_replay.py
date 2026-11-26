"""Replay a recorded live session through the WHOLE advisor, offline (GH #1426).

Feeds every message of a ``realgame_advisor --record-requests`` corpus through
the advisor's own message loop (:func:`realgame_advisor._serve_connected`) with
a fake bridge client in place of the socket, and writes every reply the mod
would have received: ``advice`` for each ``advice_request`` and the coach's
records. Nothing is stubbed past the socket: the fight mirror, Instinct, the
screen policies, the combat search, the coach and the live tree all run.

Two uses, both the groundwork of the Rust advisor port (#1426 step 0):

* **Golden replies.** ``--out`` writes the replies as JSONL (``.gz`` too),
  after one ``meta`` line naming the corpus and the options; ``--check
  GOLDEN`` replays with the golden's own options and compares. With
  ``--threads 1`` (the default here; the live default of 4 root-parallel trees
  is not reproducible) the combat search is seeded and the coach runs inline,
  so two replays of a corpus agree byte for byte on one machine. Across CPUs
  the nets' float outputs differ in the last digits (SIMD), and a rounded
  probability can land one unit of its 4th decimal apart, so the check
  compares numbers to a relative ``1e-5`` or an absolute ``1.5e-4`` and
  everything else exactly. A port
  must reproduce them.
* **What runs.** ``--trace DIR`` records which of the advisor's modules load
  and how many of their lines execute (``coverage``), against the static
  import closure of ``realgame_advisor``, into ``DIR/trace.json`` and a
  readable ``DIR/trace.md``.

``--brain search`` drops the recorded ``brain`` so the combat search and the
PPO screen policies answer instead of Instinct (the live corpus was played
with Instinct, which would otherwise leave the search path unexercised).

Usage::

    uv run python python/realgame_advisor_replay.py CORPUS.jsonl[.gz] \
        [--out replies.jsonl[.gz]] [--trace DIR] [--brain recorded|search|instinct] \
        [--sims N] [--boss-sims N] [--limit N]
    uv run python python/realgame_advisor_replay.py --check GOLDEN.jsonl.gz
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import logging
import math
import re
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

PYTHON_DIR = Path(__file__).resolve().parent
#: The monorepo root: goldens name their corpus relative to it.
REPO_ROOT = PYTHON_DIR.parents[1]

#: The one wall-clock value in a reply: Instinct's note says how long its net
#: took ("· 9 ms"). Masked so a replay is comparable run to run.
_MS = re.compile(r"\b\d+ ms\b")


class ReplayEnded(Exception):
    """The corpus is exhausted (or ``--limit`` reached): the loop's exit."""


def read_corpus(path: Path) -> Iterator[dict[str, Any]]:
    """Every message of a recorded corpus, in order (``.gz`` or plain)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class ReplayClient:
    """The bridge client's surface the advisor loop uses, over a corpus.

    ``receive_state`` hands out the recorded messages in order and raises
    :class:`ReplayEnded` at the end; ``send_action`` collects the replies.
    Every recorded request is answered: live, the advisor skips a request
    a newer one already superseded, which a recording cannot reproduce
    (the queue depth depended on the search's wall time).
    """

    def __init__(
        self, messages: Iterator[dict[str, Any]], brain: str = "recorded", limit: int = 0
    ) -> None:
        self._messages = messages
        self._brain = brain
        self._limit = limit
        self.requests = 0
        self.replies: list[dict[str, Any]] = []

    def receive_state(self) -> dict[str, Any]:
        for msg in self._messages:
            if msg.get("type") == "advice_request":
                if self._limit and self.requests >= self._limit:
                    break
                self.requests += 1
                if self.requests % 100 == 0:
                    print(f"replay: {self.requests} requests", file=sys.stderr, flush=True)
                if self._brain == "search":
                    msg.pop("brain", None)
                elif self._brain == "instinct":
                    msg["brain"] = "instinct"
            return msg
        raise ReplayEnded

    def newer_advice_request_pending(self, advice_for: Any = None) -> bool:
        return False

    def send_action(self, reply: dict[str, Any]) -> None:
        self.replies.append(reply)


def _scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    if isinstance(obj, str):
        return _MS.sub("<t> ms", obj)
    return obj


def replay(
    corpus: Path,
    *,
    brain: str = "recorded",
    sims: int = 4000,
    boss_sims: int = 16000,
    threads: int = 1,
    coach: bool = True,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Run ``corpus`` through the advisor loop and return the replies, in order."""
    import realgame_advisor as ra
    import realgame_ppo
    from realgame_advisor_mirror import AdvisorMirror
    from realgame_coach import InlineExecutor
    from realgame_control import ControlPlane
    from realgame_livetree import LiveRunTree
    from realgame_workbench_log import WorkbenchRunLogger
    from sts2_hint.recommender import RecommendConfig

    # A fresh Instinct per replay: its run graft is carried state.
    ra.INSTINCT = ra.InstinctAdvisor()
    ra.LAST_COMBAT.clear()
    args = argparse.Namespace(
        sims=sims,
        boss_sims=boss_sims,
        threads=threads,
        log_search_tree=0,
        coach_catchup_sims=2000,
        # No wall-clock cuts: a plan or a coach turn the clock ends early
        # depends on the machine's load, so the replies would too (GH #1784).
        coach_budget_s=math.inf,
        plan_budget_s=math.inf,
    )
    cfg = RecommendConfig(
        sims=sims,
        threads=threads,
        hidden_info=ra.HONEST_SEARCH["hidden_info"],
        c_puct=ra.HONEST_SEARCH["c_puct"],
        draw_widening_c=ra.HONEST_SEARCH["draw_widening_c"],
    )
    client = ReplayClient(read_corpus(corpus), brain=brain, limit=limit)
    with tempfile.TemporaryDirectory(prefix="advisor-replay-") as tmp:
        log = ra.AdvisorLog(Path(tmp))
        wlog = WorkbenchRunLogger(None)
        tree = LiveRunTree()
        control = ControlPlane(tree)
        control.set_client(client)
        policies = realgame_ppo.PpoPolicyCache(None, strict=False)
        mirror = AdvisorMirror(cfg, enabled=True)
        session: dict[str, Any] = {}
        grader = (
            ra.build_coach(client, wlog, args, policies, session, InlineExecutor(), tree)
            if coach
            else None
        )
        try:
            ra._serve_connected(
                client,
                cfg,
                args,
                log,
                wlog,
                policies,
                tree,
                control,
                mirror,
                session,
                grader,
            )
        except ReplayEnded:
            pass
    return [_scrub(r) for r in client.replies]


# -- golden replies ---------------------------------------------------------


def write_replies(path: Path, meta: dict[str, Any], replies: list[dict[str, Any]]) -> None:
    """``meta`` as the first line, then one reply per line (gzipped for ``.gz``)."""
    lines = [json.dumps({"meta": meta}, sort_keys=True)]
    lines += [json.dumps(r, sort_keys=True) for r in replies]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    if path.suffix == ".gz":
        data = gzip.compress(data, compresslevel=9, mtime=0)
    path.write_bytes(data)


def read_replies(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A golden file's ``meta`` and its replies."""
    rows = list(read_corpus(path))
    return rows[0]["meta"], rows[1:]


def compare_replies(
    got: list[dict[str, Any]],
    want: list[dict[str, Any]],
    rel: float = 1e-5,
    abs_tol: float = 1.5e-4,
    max_diffs: int = 20,
) -> list[str]:
    """Where ``got`` differs from ``want``: numbers to ``rel`` or ``abs_tol``
    (the advisor rounds probabilities to 4 decimals, so float noise across
    CPUs can land one unit apart there: 0.2679 vs 0.2678), the rest exactly."""
    diffs: list[str] = []

    def walk(a: Any, b: Any, where: str) -> None:
        if len(diffs) >= max_diffs:
            return
        num = (int, float)
        if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool):
            if abs(a - b) > max(rel * max(abs(a), abs(b)), abs_tol):
                diffs.append(f"{where}: {a!r} != {b!r}")
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                diffs.append(f"{where}: keys {sorted(a.keys() ^ b.keys())} differ")
            for k in a.keys() & b.keys():
                walk(a[k], b[k], f"{where}.{k}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                diffs.append(f"{where}: {len(a)} items != {len(b)}")
            for i, (x, y) in enumerate(zip(a, b, strict=False)):
                walk(x, y, f"{where}[{i}]")
        elif a != b:
            diffs.append(f"{where}: {str(a)[:120]!r} != {str(b)[:120]!r}")

    if len(got) != len(want):
        diffs.append(f"{len(got)} replies != {len(want)}")
    for i, (g, w) in enumerate(zip(got, want, strict=False)):
        walk(g, w, f"#{i} {w.get('action')} {w.get('request_id')}")
    return diffs


def check_golden(golden: Path, corpus_root: Path | None = None) -> list[str]:
    """Replay a golden's corpus with its options; the differences from it."""
    meta, want = read_replies(golden)
    corpus = Path(meta["corpus"])
    if not corpus.is_absolute():
        corpus = (corpus_root or REPO_ROOT) / corpus
    got = replay(corpus, **meta["options"])
    return compare_replies(json.loads(json.dumps(got)), want)


# -- what runs -----------------------------------------------------------


def _module_file(name: str) -> Path | None:
    parts = name.split(".")
    for cand in (
        PYTHON_DIR.joinpath(*parts).with_suffix(".py"),
        PYTHON_DIR.joinpath(*parts, "__init__.py"),
    ):
        if cand.is_file():
            return cand
    return None


def _module_name(path: Path) -> str:
    rel = path.relative_to(PYTHON_DIR).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def static_closure(root: str = "realgame_advisor") -> dict[str, Path]:
    """Every repo module reachable by any import statement from ``root``.

    Counts lazy (function-level) imports and ``TYPE_CHECKING`` blocks too:
    this is the upper bound the issue's "75 modules / ~44k lines" measured.
    """
    seen: dict[str, Path] = {}
    todo = [root]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        path = _module_file(name)
        if path is None:
            continue
        seen[name] = path
        todo.extend(target for target, _kind in module_imports(name, path))
        # A parent package runs its __init__ on any submodule import.
        todo.extend(_parents(name))
    return seen


def module_imports(name: str, path: Path) -> list[tuple[str, str]]:
    """The repo modules ``name`` imports, each with how: ``top`` (at import
    time), ``lazy`` (inside a function) or ``typing`` (``TYPE_CHECKING``
    only, never at run time). Names that are not repo modules are dropped.
    """
    pkg = name if path.name == "__init__.py" else name.rpartition(".")[0]
    out: list[tuple[str, str]] = []

    def visit(node: ast.AST, kind: str) -> None:
        for child in ast.iter_child_nodes(node):
            k = kind
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                k = "lazy" if kind == "top" else kind
            elif isinstance(child, ast.If) and "TYPE_CHECKING" in ast.unparse(child.test):
                for sub in child.body:
                    visit_stmt(sub, "typing")
                for sub in child.orelse:
                    visit_stmt(sub, kind)
                continue
            visit_stmt(child, k)

    def visit_stmt(node: ast.AST, kind: str) -> None:
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".") if pkg else []
                if node.level > 1:
                    base = base[: len(base) - (node.level - 1)]
                mod = ".".join([*base, node.module] if node.module else base)
            else:
                mod = node.module or ""
            # `from pkg import submodule` imports the submodule too.
            targets = [mod, *(f"{mod}.{a.name}" for a in node.names)]
        out.extend((t, kind) for t in targets if _module_file(t) is not None)
        visit(node, kind)

    visit(ast.parse(path.read_text(encoding="utf-8")), "top")
    return out


def importers(closure: dict[str, Path]) -> dict[str, list[tuple[str, str]]]:
    """For each module in ``closure``: who imports it, and how (see
    :func:`module_imports`)."""
    out: dict[str, list[tuple[str, str]]] = {name: [] for name in closure}
    for name, path in closure.items():
        for target, kind in module_imports(name, path):
            if target in out and target != name and (name, kind) not in out[target]:
                out[target].append((name, kind))
    return out


def _parents(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def _line_count(path: Path) -> int:
    return sum(1 for _ in path.open(encoding="utf-8"))


def traced_replay(
    corpora: list[Path], out_dir: Path, brains: tuple[str, ...] = ("recorded",), **kw: Any
) -> dict[str, Any]:
    """Replay every corpus with every brain under ``coverage``; write the report."""
    import coverage

    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(sys.modules)
    cov = coverage.Coverage(
        data_file=None, source=[str(PYTHON_DIR)], config_file=False, branch=False
    )
    cov.start()
    t0 = time.monotonic()
    replies = 0
    try:
        import realgame_advisor  # count its import-time code

        for brain in brains:
            for corpus in corpora:
                replies += len(replay(corpus, brain=brain, **kw))
    finally:
        cov.stop()
    elapsed = time.monotonic() - t0
    loaded = {
        _module_name(Path(f).resolve())
        for name, m in list(sys.modules.items())
        if name not in before
        and (f := getattr(m, "__file__", None))
        and Path(f).resolve().is_relative_to(PYTHON_DIR)
    }
    executed: dict[str, int] = {}
    statements: dict[str, int] = {}
    for f in cov.get_data().measured_files():
        p = Path(f).resolve()
        if not p.is_relative_to(PYTHON_DIR):
            continue
        name = _module_name(p)
        _fname, stmts, _excluded, missing, _text = cov.analysis2(f)
        statements[name] = len(stmts)
        executed[name] = len(stmts) - len(missing)
    closure = static_closure()
    who = importers(closure)
    rows = []
    for name in sorted(set(closure) | loaded):
        path = closure.get(name) or _module_file(name)
        rows.append(
            {
                "module": name,
                "lines": _line_count(path) if path else None,
                "in_static_closure": name in closure,
                "loaded": name in loaded,
                "statements": statements.get(name, 0),
                "executed": executed.get(name, 0),
                "imported_by": [f"{m} ({k})" for m, k in who.get(name, [])],
            }
        )
    report = {
        "corpora": [str(c) for c in corpora],
        "options": {"brains": list(brains), **kw},
        "replies": replies,
        "elapsed_s": round(elapsed, 1),
        "modules": rows,
    }
    (out_dir / "trace.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out_dir / "trace.md").write_text(render_trace(report), encoding="utf-8")
    return report


def render_trace(report: dict[str, Any]) -> str:
    """The module report as a markdown table, the most-run first."""
    rows = report["modules"]
    closure = [r for r in rows if r["in_static_closure"]]
    ran = [r for r in rows if r["executed"]]
    never = [r for r in closure if not r["loaded"]]
    loaded_idle = [r for r in rows if r["loaded"] and not r["executed"]]

    def total(rs: list[dict[str, Any]]) -> int:
        return sum(r["lines"] or 0 for r in rs)

    out = [
        "# Advisor replay trace",
        "",
        f"Corpora: {', '.join(report['corpora'])}  ",
        f"Options: `{json.dumps(report['options'])}`  ",
        f"Replies: {report['replies']}, {report['elapsed_s']} s under coverage",
        "",
        f"- static import closure of `realgame_advisor`: {len(closure)} modules, "
        f"{total(closure)} lines",
        f"- loaded during the replay: {sum(r['loaded'] for r in rows)} modules",
        f"- executed at least one line: {len(ran)} modules, {total(ran)} lines; "
        f"{sum(r['executed'] for r in ran)} of {sum(r['statements'] for r in ran)} "
        "statements ran",
        f"- in the closure but never loaded: {len(never)} modules, {total(never)} lines",
        "",
        "| module | lines | closure | loaded | statements run | % |",
        "|---|---:|:-:|:-:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: (-r["executed"], r["module"])):
        pct = f"{100 * r['executed'] / r['statements']:.0f}" if r["statements"] else "-"
        out.append(
            f"| `{r['module']}` | {r['lines'] or '-'} | {'y' if r['in_static_closure'] else ''}"
            f" | {'y' if r['loaded'] else ''} | {r['executed']}/{r['statements']} | {pct} |"
        )
    if never:
        out += [
            "",
            "## Reachable but never loaded",
            "",
            "Who imports each one: `top` = at import time, `lazy` = inside a function, "
            "`typing` = `TYPE_CHECKING` only.",
            "",
            "| module | lines | imported by |",
            "|---|---:|---|",
        ]
        out += [
            f"| `{r['module']}` | {r['lines']} | {', '.join(r['imported_by'])} |"
            for r in sorted(never, key=lambda r: -(r["lines"] or 0))
        ]
    if loaded_idle:
        out += ["", "Loaded but no line counted (namespace packages, data-only):", ""]
        out += [f"- `{r['module']}`" for r in loaded_idle]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("corpus", type=Path, nargs="*")
    ap.add_argument("--out", type=Path, help="write the replies here (JSONL, .gz ok)")
    ap.add_argument(
        "--check", type=Path, metavar="GOLDEN", help="replay a golden's corpus and compare"
    )
    ap.add_argument("--trace", type=Path, metavar="DIR", help="write the module trace here")
    ap.add_argument(
        "--brain",
        choices=("recorded", "search", "instinct"),
        action="append",
        help="who answers (repeatable with --trace; default: as recorded)",
    )
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--boss-sims", type=int, default=16000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-coach", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N advice requests")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.check:
        diffs = check_golden(args.check)
        for d in diffs:
            print(d)
        print(f"{args.check}: {'MATCH' if not diffs else f'{len(diffs)}+ differences'}")
        return 1 if diffs else 0
    if not args.corpus:
        ap.error("name a corpus (or --check GOLDEN)")
    brains = tuple(args.brain or ["recorded"])
    kw = {
        "sims": args.sims,
        "boss_sims": args.boss_sims,
        "threads": args.threads,
        "coach": not args.no_coach,
        "limit": args.limit,
    }
    if args.trace:
        report = traced_replay(args.corpus, args.trace, brains, **kw)
        print(render_trace(report).split("| module")[0])
        if not args.out:
            return 0
    if args.out and (len(brains) > 1 or len(args.corpus) > 1):
        ap.error("--out takes one --brain and one corpus")
    replies = [r for b in brains for c in args.corpus for r in replay(c, brain=b, **kw)]
    errors = sum("advisor error" in str((r.get("advice") or {}).get("note") or "") for r in replies)
    if args.out:
        corpus = args.corpus[0].resolve()
        name = corpus.relative_to(REPO_ROOT) if corpus.is_relative_to(REPO_ROOT) else corpus
        meta = {"corpus": name.as_posix(), "options": {"brain": brains[0], **kw}}
        write_replies(args.out, meta, replies)
    print(f"{len(replies)} replies, {errors} advisor errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
