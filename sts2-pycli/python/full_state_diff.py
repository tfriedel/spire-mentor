"""Full-state combat diff for the act2/3 run-parity driver.

``act23_state_parity`` compares a coarse run summary per step (phase, HP,
gold, deck multiset, relics, potion count, enemy HP), so a desync is flagged
only once it reaches HP, often dozens of steps after the real split. The
splits behind GH #1054 item 7 were a hand-order difference, a random stream one
read ahead, or a play counter: none of those show up in HP until much later.

This module compares far more of the combat state at every combat step,
building on the parity fuzzer's canonical projection
(:mod:`parity_fuzz_state`): player and monster HP, block and powers, orbs,
Osty, relics, the turn counter. On top of that it compares:

* the hand, draw, discard and exhaust piles **in order** (a shuffle or
  hand-order split is visible at once),
* each hand card's resolved energy cost,
* each hand card's affliction and its amount (a Bound split is visible on
  the step it happens, not at the later ``illegal_play`` reject; GH #1150),
* every RNG stream cursor both engines report (a stream one read ahead is
  visible even when the value it produced happened to match).

Monster move-state ids are not compared: the engines name moves differently
(C# ``SHRINKER_MOVE``, Rust ``SHRINK``).

The C# side reads the oracle's ``get_state`` command
(``RunSimulator.GetStateImpl``), the Rust side ``sts2_rs.state_to_dict``.
This is the oracle half of GH #739 (field-by-field checkpoints like the
game's ``NetFullCombatState``), without the real-game captures.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from parity_audit import _canon_card
from parity_fuzz_state import Divergence, diff, normalize_csharp, normalize_rust


def canon_card_id(raw: Any) -> str | None:
    """One card token for both engines: C# ``CARD.RUPTURE`` and Rust
    ``RUPTURE_CARD`` both become ``RUPTURE`` (``parity_audit._canon_card``)."""
    return None if raw is None else _canon_card(str(raw))


#: Rust monster kinds that are engine-local variants of one C# monster class:
#: the Rust port splits the Dense Vegetation ``Wriggler`` into its own kind
#: (per-slot AI), while C# spawns the ordinary ``Wriggler``.
RUST_MONSTER_KIND_ALIASES: dict[str, str] = {
    "DENSE_VEGETATION_WRIGGLER": "WRIGGLER",
}


def _rust_card_cost(card: dict[str, Any]) -> int | None:
    """Resolved energy cost, or ``None`` for an X-cost card (``base_cost`` -1):
    C# reports an X card's resolved cost as the current energy, which is not
    a property of the card."""
    if card.get("base_cost") == -1:
        return None
    over = card.get("cost_override")
    return int(over if over is not None else card.get("base_cost", 0))


def project_csharp(get_state: dict[str, Any]) -> dict[str, Any]:
    """Canonical full-state dict from a C# ``get_state`` response."""
    out = normalize_csharp(get_state)
    out["piles"] = {
        pile: [canon_card_id(c) for c in (get_state.get(f"{pile}_ids") or [])]
        for pile in ("hand", "draw", "discard", "exhaust")
    }
    out["hand_costs"] = [int(c) for c in (get_state.get("hand_costs") or [])]
    affs = get_state.get("hand_afflictions")
    amounts = get_state.get("hand_affliction_amounts") or []
    # ``None`` = an oracle build that predates the field: not compared.
    out["hand_afflictions"] = (
        None
        if affs is None
        else [
            _canon_affliction(a, amounts[i] if i < len(amounts) else 0) for i, a in enumerate(affs)
        ]
    )
    out["rng"] = {k: int(v) for k, v in (get_state.get("rng_counters") or {}).items()}
    return _drop_uncompared(out)


def project_rust(combat_state: Any) -> dict[str, Any]:
    """Canonical full-state dict from a native ``sts2_rs`` combat state."""
    import sts2_rs

    return project_rust_dict(sts2_rs.state_to_dict(combat_state))


def project_rust_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Canonical full-state dict from a ``sts2_rs.state_to_dict`` dict."""
    out = normalize_rust(SimpleNamespace(state_to_dict=lambda: d))
    for m in out["monsters"]:
        m["kind"] = RUST_MONSTER_KIND_ALIASES.get(m["kind"], m["kind"])
    deck = d.get("deck") or {}
    out["piles"] = {
        pile: [canon_card_id(c.get("id")) for c in (deck.get(key) or [])]
        for pile, key in (
            ("hand", "hand"),
            ("draw", "draw_pile"),
            ("discard", "discard_pile"),
            ("exhaust", "exhaust_pile"),
        )
    }
    out["hand_costs"] = [_rust_card_cost(c) for c in (deck.get("hand") or [])]
    out["hand_afflictions"] = [
        _canon_affliction(c.get("affliction"), c.get("affliction_stacks") or 0)
        for c in (deck.get("hand") or [])
    ]
    out["rng"] = {k: int(v) for k, v in ((d.get("rng") or {}).get("counters") or {}).items()}
    return _drop_uncompared(out)


def _canon_affliction(raw: Any, amount: Any) -> dict[str, Any] | None:
    """One affliction token for both engines: C# reports the model id entry
    (``BOUND``), Rust the type name (``Bound``)."""
    if not raw:
        return None
    return {"id": str(raw).upper(), "amount": int(amount or 0)}


def _drop_uncompared(out: dict[str, Any]) -> dict[str, Any]:
    """Fields the canonical projection carries that this diff leaves out.

    ``relics``: the Rust combat state omits run relics with no combat effect
    (a Neow ``NEOWS_SACRIFICE`` is on the run but not in the fight), and the
    coarse run diff already compares the run's relics.
    """
    out.pop("relics", None)
    return out


def diff_full(cs: dict[str, Any], rs: dict[str, Any]) -> list[Divergence]:
    """Field-by-field divergences between two projected states.

    RNG streams are compared only where BOTH engines report the stream: the
    Rust dump carries a few engine-local cursors (and C# a few the Rust
    combat state does not track), and a missing key is a reporting gap, not
    a desync. A hand position holding an X-cost card (``None`` on the Rust
    side) skips the cost check. Hand afflictions are compared only when the
    C# ``get_state`` reports them (older oracle builds do not).
    """
    shared = cs["rng"].keys() & rs["rng"].keys()
    if cs.get("hand_afflictions") is None:
        cs = {k: v for k, v in cs.items() if k != "hand_afflictions"}
        rs = {k: v for k, v in rs.items() if k != "hand_afflictions"}
    cs_costs = list(cs["hand_costs"])
    if len(cs_costs) == len(rs["hand_costs"]):
        cs_costs = [None if r is None else c for c, r in zip(cs_costs, rs["hand_costs"])]
    cs = {**cs, "rng": {k: cs["rng"][k] for k in shared}, "hand_costs": cs_costs}
    rs = {**rs, "rng": {k: rs["rng"][k] for k in shared}}
    return diff(cs, rs)


def combat_split_lines(
    cs_phase: str, rs_phase: str, full_lines: list[str], headline_fields: list[str]
) -> list[str]:
    """The full-state split for a step where either engine is in combat.

    ``full_lines`` is the full-state diff (only computed while both engines
    are in combat), ``headline_fields`` the coarse per-step diff of the same
    step (``run_parity.diff_snaps``). The projection alone misses a step where
    one engine has left the combat and the other has not (Regent v0.107.1
    seed 6488: C# still ``combat``, Rust at the ``card_reward``, reported as
    ``full=clean``), and it covers only combat fields. So the full-state check
    stays a superset of the coarse one: on a phase disagreement, or when the
    projection matched but the coarse diff did not, the coarse fields are the
    split.
    """
    if cs_phase == rs_phase and full_lines:
        return full_lines
    return list(headline_fields)


def path_class(path: str) -> str:
    """``piles.hand[3]`` → ``piles.hand``; ``monsters[0].powers[1].amount`` →
    ``monsters.powers.amount``. Groups first splits across runs by field."""
    out, depth = [], 0
    for ch in path:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out).removesuffix(".len")
