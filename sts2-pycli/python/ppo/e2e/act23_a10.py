"""F4 at A10: ``act23_state_parity`` with the second boss compared as a meta node.

An A10 run's final act has a second boss node above the act boss. The C#
headless reports it only as the ``second_boss`` meta node of ``get_map``
(RunSimulator.cs, "A10 DoubleBoss"), while Rust's ``full_map`` also lists it
in ``rows``. ``run_parity._norm_map_graph`` reads ``rows`` and ``boss`` only,
so every A10 run that reaches act 3 stops there on ``map.nodes:
only_rs=[(3, 15)]`` and act 3 goes unchecked. This wrapper drops that row
node and compares a ``BOSS2`` entry built from ``second_boss`` on both sides,
then runs the unchanged driver. Fork workers inherit the patch; spawn and
forkserver workers re-import this module as ``__mp_main__``, which applies it
again (see the bottom of the file).

Run it from ``sts2-pycli/`` with ``act23_state_parity.py``'s arguments::

    PYTHONPATH=python python -m ppo.e2e.act23_a10 --character Ironclad \\
        --ascension 10 --game-version v0.111.0 --full-state --trace-split -n 50 --workers 16
"""

from __future__ import annotations

import sys
from typing import Any

import run_parity

_norm_map_graph = run_parity._norm_map_graph


def norm_map_graph_a10(m: dict[str, Any]) -> dict[Any, Any]:
    """``run_parity._norm_map_graph`` with the second boss as a ``BOSS2`` entry."""
    nodes = _norm_map_graph(m)
    second = m.get("second_boss")
    if isinstance(second, dict) and second.get("row") is not None:
        coord = (second.get("col"), second.get("row"))
        nodes.pop(coord, None)
        nodes["BOSS2"] = (run_parity._canon(second.get("type") or "?"), coord)
    return nodes


def install() -> None:
    """Point ``run_parity`` at :func:`norm_map_graph_a10` in this process."""
    run_parity._norm_map_graph = norm_map_graph_a10


def main(argv: list[str] | None = None) -> int:
    install()
    import act23_state_parity

    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    return act23_state_parity.main()


# ``act23_state_parity`` starts its workers with the default start method. A
# spawn or forkserver worker runs this module again under the name
# ``__mp_main__`` (``python -m`` passes the module name on), so it patches its
# own ``run_parity`` here. A plain import (the tests) patches nothing.
if __name__ == "__mp_main__":
    install()

if __name__ == "__main__":
    sys.exit(main())
