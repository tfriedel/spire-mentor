"""Explain a full-state combat split with the C# oracle's state trace (GH #1088).

The full-state diff (:mod:`full_state_diff`) names the step and the fields
where the engines first disagree. One step can be a whole card play with
dozens of hooks inside it, so the field alone rarely names the culprit.

The oracle's ``trace`` command (``Sts2Headless/StateTrace.cs``) records every
C# write to creature HP / max HP / block, power amounts, combat piles, energy
/ stars and the named RNG streams, each with ``src``: the model methods
(hook overrides, ``OnPlay``, monster moves) active at the time. Every write
passes through a private setter or an ``*Internal`` method, so the C# history
is complete. The Rust engine is not traced; its writes are spread over
hundreds of sites. Instead each diverging field gets its C# change history
next to the Rust end value::

    monsters[0].hp: cs=90 rs=80
      [812 play_card(card_index=2)] 100 -> 94  StrikeIronclad.OnPlay
      [812 play_card(card_index=2)] 94 -> 90   Hellraiser.AfterCardDrawnEarly>StrikeIronclad.OnPlay

The step started equal on both sides (the previous check was clean), so the
C# change that Rust did not make, or made differently, is the lead.
"""

from __future__ import annotations

import re
from typing import Any

from full_state_diff import canon_card_id

#: C# ``PileType`` names -> the full-state diff's pile keys.
_PILES = {
    "Hand": "hand",
    "Draw": "draw",
    "Discard": "discard",
    "Exhaust": "exhaust",
    "Play": "play",
}

#: Most history lines printed per diverging field.
MAX_LINES_PER_FIELD = 12

_CREATURE = re.compile(r"^(player|osty|monsters\[\d+\])\.(\w+)")


def _field_filter(path: str) -> tuple[str, Any] | None:
    """What a diff path's history is made of: ``(label, predicate)``.

    The label groups paths that share a history (``piles.discard[3]`` and
    ``discard_count`` both read the discard pile's events).
    """
    m = _CREATURE.match(path)
    if m:
        who, attr = m.group(1), m.group(2)
        if attr == "powers":
            return (
                f"{who}.powers",
                lambda e: e["k"] in ("power", "power_remove") and e.get("who") == who,
            )
        kind = "hp" if attr == "alive" else attr
        return f"{who}.{kind}", lambda e: e["k"] == kind and e.get("who") == who
    for cs_name, key in _PILES.items():
        if path.startswith((f"piles.{key}", f"{key}_count")) or (
            key == "hand" and path.startswith(("hand_costs", "hand_afflictions"))
        ):
            return (
                f"piles.{key}",
                lambda e, p=cs_name: e.get("pile") == p and e["k"].startswith("pile_"),
            )
    if path.startswith("rng."):
        stream = path.split(".", 1)[1]
        return path, lambda e: e["k"] == "rng" and e.get("who") == stream
    return None


def _describe(e: dict[str, Any]) -> str:
    k = e["k"]
    if k == "power":
        return f"{e.get('id')} {e.get('old')} -> {e.get('new')}"
    if k == "power_remove":
        return f"{e.get('id')} removed (was {e.get('old')})"
    if k == "pile_add":
        return f"+{canon_card_id(e.get('id'))} @{e.get('idx')}"
    if k == "pile_remove":
        return f"-{canon_card_id(e.get('id'))} @{e.get('idx')}"
    if k == "pile_clear":
        return f"cleared ({e.get('old')} cards)"
    if k == "rng":
        return f"draw -> cursor {e.get('new')}"
    return f"{e.get('old')} -> {e.get('new')}"


def _collapse_rng(rows: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any], int]]:
    """Merge runs of draws on one stream with one source into a single row."""
    out: list[tuple[str, dict[str, Any], int]] = []
    for seg, e in rows:
        if (
            out
            and e["k"] == "rng"
            and out[-1][1]["k"] == "rng"
            and out[-1][0] == seg
            and out[-1][1].get("src") == e.get("src")
        ):
            first = out[-1]
            out[-1] = (seg, {**e, "first": first[1].get("first", first[1]["new"])}, first[2] + 1)
        else:
            out.append((seg, e, 1))
    return out


def flatten(segments: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """``(segment label, event)`` pairs in order."""
    rows = []
    for s in segments:
        label = f"{s.get('seq')} {s.get('action')}({s.get('args') or ''})"
        rows.extend((label, e) for e in s.get("events") or [])
    return rows


def explain(segments: list[dict[str, Any]], divs: list[dict[str, Any]]) -> list[str]:
    """Report lines: each diverging field's C# history since the last clean check."""
    rows = flatten(segments)
    out: list[str] = []
    seen: set[str] = set()
    for d in divs:
        f = _field_filter(d["path"])
        head = f"{d['path']}: cs={d['csharp']!r} rs={d['rust']!r}"
        if f is None:
            out.append(f"{head}  (not traced)")
            continue
        label, pred = f
        if label in seen:
            continue
        seen.add(label)
        out.append(head)
        hist = _collapse_rng([(seg, e) for seg, e in rows if pred(e)])
        if not hist:
            out.append("  (no C# change this step: Rust changed it, or C# skipped a change)")
            continue
        for seg, e, n in hist[:MAX_LINES_PER_FIELD]:
            what = _describe(e)
            if n > 1:
                what = f"{n} draws -> cursor {e.get('first')}..{e['new']}"
            out.append(f"  [{seg}] {what}  {e.get('src') or '(engine)'}")
        if len(hist) > MAX_LINES_PER_FIELD:
            out.append(f"  ... {len(hist) - MAX_LINES_PER_FIELD} more")
    # Everything that acted in C#: when Rust made a change C# did not, the
    # Rust culprit is usually one of these hooks firing where C#'s did not.
    acted = list(dict.fromkeys(e["src"] for _, e in rows if e.get("src")))
    if acted:
        out.append(f"C# sources this step: {', '.join(acted[:MAX_LINES_PER_FIELD])}")
    return out


def suspects(
    segments: list[dict[str, Any]], divs: list[dict[str, Any]], limit: int = 3
) -> list[str]:
    """Distinct innermost sources of the diverging fields' C# changes, in order."""
    rows = flatten(segments)
    preds = [f[1] for d in divs if (f := _field_filter(d["path"])) is not None]
    out: list[str] = []
    for _, e in rows:
        src = (e.get("src") or "").rsplit(">", 1)[-1]
        if src and src not in out and any(p(e) for p in preds):
            out.append(src)
            if len(out) == limit:
                break
    return out
