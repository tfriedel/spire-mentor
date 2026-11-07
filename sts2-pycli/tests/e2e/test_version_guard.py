"""AST guard: every engine call on the e2e path names its game build (WP1).

Rust never reads `STS2_GAME_VERSION`, so `sts2_rs.RunManager(...)` or
`sts2_rs.build_scenario_state(...)` without `game_version=` silently simulates
v0.107.1. This test parses every module of the e2e path and fails on such a
call. A call routed through `sts2_client.rs_engine_call` (which always passes
the kwarg) or through `ppo.e2e.factory` counts as versioned. A bare reference
to either constructor (`make = sts2_rs.RunManager`) or an import alias fails
too: it would let an unversioned call through under another name.
"""

from __future__ import annotations

import ast
from pathlib import Path

PPO = Path(__file__).resolve().parents[2] / "python" / "ppo"
# Plan section 3.1: the guarded part of `ppo/`.
GUARDED = ("e2e", "combat_ppo", "rollout_starts.py", "combat_bench")
TARGETS = frozenset({"RunManager", "build_scenario_state"})
# Calls that pass `game_version=` for their first argument.
ROUTERS = frozenset({"rs_engine_call"})


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _annotation_ids(tree: ast.AST) -> set[int]:
    """`id()` of every node inside a type annotation (a reference, not a use)."""
    anns: list[ast.expr | None] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            a = node.args
            args = [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]
            anns += [node.returns, *(x.annotation for x in args if x is not None)]
        elif isinstance(node, ast.AnnAssign):
            anns.append(node.annotation)
    return {id(n) for ann in anns if ann is not None for n in ast.walk(ann)}


def unversioned_calls(source: str, filename: str = "<snippet>") -> tuple[list[str], int]:
    """`(violations, versioned)` for one module's source.

    `violations` holds one `file:line: what` per engine call without an
    explicit `game_version=` (a `**kwargs` splat does not count, nor does
    `game_version=None`), per bare reference to an engine constructor that is
    not handed to a router, and per import alias of one. `versioned` counts
    the engine calls that pass.
    """
    tree = ast.parse(source, filename=filename)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    callees = {id(c.func) for c in calls}
    routed = {id(c.args[0]) for c in calls if _name(c.func) in ROUTERS and c.args}
    skip = callees | routed | _annotation_ids(tree)
    out: list[str] = []
    versioned = 0
    for c in calls:
        name = _name(c.func)
        if name not in TARGETS:
            continue
        kw = next((k for k in c.keywords if k.arg == "game_version"), None)
        if kw is None:
            out.append(f"{filename}:{c.lineno}: {name}(...) without game_version=")
        elif isinstance(kw.value, ast.Constant) and kw.value.value is None:
            out.append(f"{filename}:{c.lineno}: {name}(..., game_version=None)")
        else:
            versioned += 1
    for node in ast.walk(tree):
        if isinstance(node, ast.Name | ast.Attribute) and _name(node) in TARGETS:
            if id(node) not in skip:
                out.append(f"{filename}:{node.lineno}: bare reference to {_name(node)}")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in TARGETS and alias.asname:
                    out.append(f"{filename}:{node.lineno}: {alias.name} imported as {alias.asname}")
    return out, versioned


def guarded_files() -> list[Path]:
    files: list[Path] = []
    for entry in GUARDED:
        path = PPO / entry
        files += [path] if path.suffix == ".py" else sorted(path.rglob("*.py"))
    return files


def test_guarded_modules_name_their_build() -> None:
    files = guarded_files()
    violations: list[str] = []
    versioned = 0
    for path in files:
        bad, ok = unversioned_calls(path.read_text(encoding="utf-8"), str(path.relative_to(PPO)))
        violations += bad
        versioned += ok
    assert not violations, "\n".join(violations)
    # Not vacuous: the scan reached the factory, build_fight and the bench.
    assert {"factory.py", "rollout_starts.py", "run.py"} <= {p.name for p in files}
    assert versioned >= 10


def test_checker_flags_unversioned_snippets() -> None:
    bad = {
        "sts2_rs.RunManager(1)": "without game_version=",
        "RunManager(seed=1, character='Ironclad')": "without game_version=",
        "sts2_rs.build_scenario_state(1, 'Ironclad', [], 50, 80, [], 'X')": "without",
        "sts2_rs.RunManager(**kwargs)": "without game_version=",
        "sts2_rs.RunManager(1, game_version=None)": "game_version=None",
        "make = sts2_rs.RunManager": "bare reference",
        "functools.partial(sts2_rs.build_scenario_state, 1)": "bare reference",
        "from sts2_rs import RunManager as RM": "imported as RM",
    }
    for snippet, why in bad.items():
        out, _ = unversioned_calls(snippet)
        assert len(out) == 1 and why in out[0], (snippet, out)


def test_checker_passes_versioned_snippets() -> None:
    good = [
        "sts2_rs.RunManager(1, game_version='v0.111.0')",
        "sts2_rs.RunManager(seed=1, game_version=ver, **kwargs)",
        "sts2_rs.build_scenario_state(*a, game_version=sts2_client.GAME_VERSION)",
        "sts2_client.rs_engine_call(sts2_rs.RunManager, seed=1, version=v)",
        "rs_engine_call(sts2_rs.build_scenario_state, 1, 'Ironclad')",
        "factory.new_run(5)",
        "def f(rm: sts2_rs.RunManager) -> sts2_rs.RunManager: ...",
        "from sts2_rs import RunManager",
    ]
    for snippet in good:
        assert unversioned_calls(snippet)[0] == [], snippet


def test_guard_goes_red_on_an_injected_call() -> None:
    path = PPO / "rollout_starts.py"
    source = path.read_text(encoding="utf-8")
    assert unversioned_calls(source)[0] == []
    injected = (
        source
        + "\n\ndef _injected():\n    import sts2_rs\n\n    return sts2_rs.RunManager(seed=1)\n"
    )
    out, _ = unversioned_calls(injected, "rollout_starts.py")
    line = len(source.splitlines()) + 6
    assert out == [f"rollout_starts.py:{line}: RunManager(...) without game_version="]
    # Dropping the kwarg from a real call goes red too.
    stripped = source.replace("        game_version=ver,\n", "", 1)
    assert stripped != source
    assert len(unversioned_calls(stripped)[0]) == 1
