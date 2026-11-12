"""Reviewed fingerprints of every game method the headless host overrides (GH #1299).

The host skips some game methods and runs something else instead: a Harmony
prefix that returns ``bool`` (a copy of the rules, or a no-op on the claim that
the method is presentation-only), or an IlPatcher rewrite baked into the
vendored ``sts2.dll``. Such an override keeps the rules of the build it was
written against. ``data/host_overrides.json`` records, per game version, a hash
of each overridden original as it was when someone last checked the override
still holds. ``tests/test_host_override_fingerprints.py`` fails when a hash
moves, or when a method joins or leaves the set.

When that test fails, for each listed method:

1. Read the new original in ``sts2-wiki/decompiled/<version>/`` and the host's
   override (grep ``RunSimulator.cs`` for the prefix the test names).
2. If the override no longer matches the game, fix or delete it. A copy of the
   rules should usually go: run the original if it now runs headless (GH #1285).
3. Then refresh the table::

       uv run python python/host_override_fingerprints.py --write --version v0.111.0

Harmony overrides are hashed from the IL the host loads (``override_fingerprints``
host command). The IlPatcher rewrites replace the body inside the vendored DLL, so
the loaded IL is already the rewrite; those are hashed from the decompiled source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

PYCLI = Path(__file__).resolve().parent.parent
TABLE = PYCLI / "data" / "host_overrides.json"
DECOMP = PYCLI.parent / "sts2-wiki" / "decompiled"

# Methods tools/IlPatcher rewrites in sts2.dll: (decomp file, method name).
CECIL_TARGETS: list[tuple[str, str]] = [
    (
        "MegaCrit.Sts2.Core.Combat/CombatManager.cs",
        "WaitUntilQueueIsEmptyOrWaitingOnNonPlayerDrivenAction",
    ),
]

_MODIFIER = (
    r"(?:public|private|protected|internal|static|async|override|virtual|sealed|new|extern|unsafe)"
)


def load_table() -> dict[str, Any]:
    return json.loads(TABLE.read_text(encoding="utf-8"))


def decomp_method_source(version: str, rel: str, name: str) -> str:
    """The declaration and brace-matched body of ``name`` in a decompiled file."""
    lines = (DECOMP / version / rel).read_text(encoding="utf-8").splitlines()
    decl = re.compile(
        rf"^\s*{_MODIFIER}(?:\s+{_MODIFIER})*\s+[\w<>\[\],.? ]+\s+{re.escape(name)}\s*\("
    )
    hits = [i for i, line in enumerate(lines) if decl.match(line)]
    if len(hits) != 1:
        raise LookupError(f"{rel}: expected one declaration of {name}, found {len(hits)}")
    out: list[str] = []
    depth = 0
    opened = False
    for line in lines[hits[0] :]:
        out.append(line.strip())
        depth += line.count("{") - line.count("}")
        opened = opened or "{" in line
        if opened and depth == 0:
            return "\n".join(out)
    raise LookupError(f"{rel}: unbalanced braces after {name}")


def cecil_fingerprints(version: str) -> dict[str, str]:
    return {
        f"{rel}::{name}": hashlib.sha256(
            decomp_method_source(version, rel, name).encode("utf-8")
        ).hexdigest()[:16]
        for rel, name in CECIL_TARGETS
    }


def harmony_fingerprints(game: Any) -> dict[str, dict[str, Any]]:
    """``{method: {hash, prefixes}}`` from a live host; patches install at run start."""
    game.start(character="Ironclad", seed="FINGERPRINT")
    reply = game.send({"cmd": "override_fingerprints"})
    if reply.get("type") != "override_fingerprints":
        raise RuntimeError(f"override_fingerprints failed: {reply}")
    return {r["method"]: {"hash": r["hash"], "prefixes": r["prefixes"]} for r in reply["overrides"]}


def diff(reviewed: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """One line per method whose fingerprint differs from the reviewed one."""
    out = []
    for key in sorted(reviewed.keys() | current.keys()):
        old, new = reviewed.get(key), current.get(key)
        if old == new:
            continue
        if old is None:
            out.append(f"  NEW      {key}  {new}")
        elif new is None:
            out.append(f"  GONE     {key}")
        else:
            out.append(f"  CHANGED  {key}  {old} -> {new}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--version", action="append", help="game version(s); default: both")
    parser.add_argument("--write", action="store_true", help="record the current fingerprints")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(PYCLI / "python"))
    import sts2_client

    versions = args.version or list(sts2_client.SUPPORTED_GAME_VERSIONS)
    table = load_table() if TABLE.exists() else {}
    status = 0
    for version in versions:
        sts2_client.set_game_version(version)
        game = sts2_client.Game()
        try:
            harmony = {k: v["hash"] for k, v in harmony_fingerprints(game).items()}
        finally:
            game.close()
        current = {"harmony": harmony, "cecil": cecil_fingerprints(version)}
        reviewed = table.get(version, {})
        lines = [
            line
            for part in ("harmony", "cecil")
            for line in diff(reviewed.get(part, {}), current[part])
        ]
        print(
            f"{version}: {len(harmony)} Harmony overrides, {len(current['cecil'])} IlPatcher rewrites"
        )
        print("\n".join(lines) if lines else "  matches the reviewed table")
        if lines:
            status = 1
        table[version] = current
    if args.write:
        TABLE.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {TABLE.relative_to(PYCLI)}")
        return 0
    return status


if __name__ == "__main__":
    raise SystemExit(main())
