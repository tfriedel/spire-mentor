#!/usr/bin/env python3
"""Record golden HTTP fixtures of the Python workbench server for the Rust port.

The Python FastAPI workbench (``workbench.server.create_app``) is the ORACLE
for the browser-only Rust port (``sts2-rs/crates/sts2-workbench``). This script
drives the Python server in-process (``fastapi.testclient.TestClient``, no
network, rust engine backend) with a fixed scripted policy over a matrix of
scenarios, and records every request it made together with a SHA-256 of the
normalized response — plus the full normalized body for a subset of them. A
Rust nextest test replays ``requests.jsonl`` request-by-request through the
port and compares hashes (and, where stored, full bodies).

Usage (from ``sts2-pycli/``)::

    PYTHONPATH=python .venv/bin/python scripts/dump_workbench_golden.py
    PYTHONPATH=python .venv/bin/python scripts/dump_workbench_golden.py \\
        --out /tmp/golden --only 'ironclad_*'
    PYTHONPATH=python .venv/bin/python scripts/dump_workbench_golden.py --write-scenarios

Scenarios live in ``tests/workbench/golden/scenarios/*.json``; fixtures are
written to ``../sts2-rs/crates/sts2-workbench/tests/golden/<scenario>/``.

Recording format (THE CONTRACT with the Rust test)
==================================================

Per scenario directory:

* ``requests.jsonl`` — one JSON object per request, in order::

      {"i": n, "method": "GET"|"POST", "route": "/api/live/action",
       "body": {...} | null, "status": 200, "sha256": "<hex>",
       "decision": "<decision of the resulting step>" | null}

  ``i`` counts from 0 over ALL requests of the scenario (the initial
  ``POST /api/live/start`` is ``i == 0``). ``decision`` is the ``decision`` of
  the Step the response carries (top-level for start/state/load, under
  ``"step"`` for action/seek) and ``null`` for responses that carry no Step
  (map, timeline, save, saves list).

* ``responses/<i>.json`` — the FULL normalized response body, pretty-printed
  with ``json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False)`` for:
  (a) every probe request, (b) the first request of the scenario whose
  resulting ``(decision, context)`` pair had not been seen yet in that
  scenario, (c) every request whose ``status != 200``, and (d) EVERY request
  in the first ``--full-steps`` (default 40) policy steps of a scenario
  flagged ``"full": true`` (the three long scenarios).

* ``meta.json`` — the scenario parameters, the engine build
  (``sts2_rs.version()`` / ``sts2_rs.src_hash()``), the Python major.minor,
  the determinism patches applied, and the coverage tags this scenario hit.

Normalization (applied BEFORE hashing and BEFORE writing a response; the
Rust side applies the same to its own response before comparing):

1. Recursively sort object keys.
2. Drop the keys ``log_path`` and ``path`` wherever they occur (they carry
   host filesystem paths).
3. Round every float to 4 decimals (Python ``round(x, 4)``). Floats stay
   floats — an integral float such as ``1.0`` is NOT converted to an int.
   Ints, bools, strings and nulls are untouched.
4. The hash is ``sha256(json.dumps(normalized, sort_keys=True,
   separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()``
   — i.e. the most compact JSON encoding, no whitespace, keys sorted, raw
   (unescaped) non-ASCII.

Determinism
===========

* ``STS2_ASSETS_BASE_URL`` is forced to ``https://assets.test`` (read at
  import time by ``workbench.server``) so every art URL is stable.
* The rust engine backend is used (``backend: "rust"``), with seeds that are
  already canonical (uppercase, no ``O``/``I``).
* ``workbench.saves``' clock is pinned so ``saved_at`` is always
  ``2026-01-01T00:00:00Z``.
* The scenario's scratch dir (``mkdtemp``-salted) is written as ``<work>``
  wherever it appears (the quit log's path in ``/api/runs``, ``/api/runs/stat``
  and ``/api/load``), and the quit log's mtime is pinned to that clock, so
  two dumps of one scenario are byte-identical.
* The PPO champion loader (``workbench.ppo_policy._get_rust_agent``) is
  stubbed to answer "no net" with the SAME error string the browser port
  reports before its host has fetched a champion (``no PPO net loaded for
  <character>``): the panel says so, node ``value``s are ``null``, the
  ``ppo`` brain falls back to the heuristic — on both hosts. A scenario
  flagged ``"ppo": true`` lifts the stub and serves the committed champion
  through the Rust ONNX backend, which is the exact inference the port runs
  (tract, same bytes), so its panel probabilities, node values and WPA
  review hash-match too. Nothing else in the server is touched.
* The server is built with the SAME data bundle the Rust crate embeds
  (``sts2-rs/crates/sts2-workbench/data/game/{creature_scales,asset_manifest}.json``)
  and the sibling ``sts2-wiki/data`` manifests the Python loaders read.

The scripted policy
===================

Deterministic, never calls a hint/advisor route, and decides from nothing but
the previous response (see :class:`Policy`; the three long scenarios use the
``"policy": "survivor"`` variant, :class:`SurvivorPolicy`, whose greedy combat
and HEAL-when-low rest choice let a run reach the act boss — the basic policy
dies around floor 5-9 on every character). Probes (``/api/live/state``,
``/api/live/map``, ``/api/live/timeline``, ``/api/live/seek``,
``/api/live/save``, ``/api/saves``, ``/api/saves/load``) are interleaved at
fixed step numbers from the scenario's ``probes`` list. A ``seek`` probe first
GETs the timeline, then seeks ``back`` nodes up the active path; play then
continues from there, which FORKS the branch tree.

Coverage
========

After all scenarios the union of observed ``(decision, context)`` pairs and
special situations is checked against a MUST list (exit status 1 if any is
missing) and a SHOULD list (reported only). See :data:`MUST` / :data:`SHOULD`.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The art origin every recorded URL is stamped with. Applied by
#: :func:`apply_determinism_patches` (never at import: the pytest that imports
#: this module shares its process with tests that mount local ``/assets``, and
#: a module-level ``os.environ`` write made ``create_app`` skip that mount).
ASSETS_BASE_URL = "https://assets.test"

_HERE = Path(__file__).resolve().parent
_PYCLI = _HERE.parent
_MONO = _PYCLI.parent
_RS_WORKBENCH_DATA = _MONO / "sts2-rs" / "crates" / "sts2-workbench" / "data" / "game"
_WIKI_DATA = _MONO / "sts2-wiki" / "data"

DEFAULT_SCENARIOS = _PYCLI / "tests" / "workbench" / "golden" / "scenarios"
DEFAULT_OUT = _MONO / "sts2-rs" / "crates" / "sts2-workbench" / "tests" / "golden"

#: Keys dropped anywhere in a response before hashing/writing (host paths).
DROP_KEYS = frozenset({"log_path", "path"})
#: What the scenario's scratch dir is written as in the fixtures.
WORK_PLACEHOLDER = "<work>"
#: The pinned clock (2026-01-01T00:00:00Z) as an epoch, for the quit log's mtime.
PINNED_EPOCH = 1767225600

#: Float precision of the normalized form.
FLOAT_DECIMALS = 4
#: ``saved_at`` every save file / listing carries.
FIXED_SAVED_AT = "2026-01-01T00:00:00Z"
#: Rule (d): full bodies for every request in the first N steps of a
#: ``"full": true`` scenario.
DEFAULT_FULL_STEPS = 40
#: Deck-level ``card_select`` contexts (everything else is an in-combat choice
#: or a choose-screen).
DECK_CONTEXTS = frozenset({"remove", "upgrade", "transform", "enchant", "grant"})
#: Map room preference, first match wins (labels look like ``"Shop (col,row)"``).
MAP_PREFERENCE = ("Shop", "RestSite", "Treasure", "Elite", "Event", "Monster", "Ancient", "Boss")

CHARACTERS = ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")
#: Already canonical: uppercase, no ``O``/``I`` (the engine maps O->0, I->1).
SEEDS = ("G0LDEN0001", "G0LDEN0002")

#: Coverage tags that MUST be hit by the union of all scenarios.
MUST: tuple[tuple[str, str], ...] = (
    ("decision:combat_play", "combat_play"),
    ("decision:map_select", "map_select"),
    ("decision:card_reward", "card_reward"),
    ("decision:shop", "shop"),
    ("decision:rest_site", "rest_site"),
    ("smith", "rest-site SMITH sub-state (card_select/upgrade after rest_site)"),
    ("card_select:remove|upgrade", "card_select with context remove OR upgrade"),
    ("combat_card_select", "in-combat card_select (context other than the deck ones)"),
    ("boss_relic", "boss relic pick (event_choice after a boss)"),
    ("room_reveal", "room_reveal present on some state"),
    ("decision:game_over", "game_over"),
)
#: Coverage tags reported but not required.
SHOULD: tuple[tuple[str, str], ...] = (
    # Unreachable on the rust backend: the engine auto-resolves a chest on
    # entry and jumps straight to the map (``RustGame._treasure_reveal``), so
    # the only treasure evidence is ``room_reveal`` (a MUST above).
    ("treasure", 'event_choice with event_name == "Treasure" (rust backend: never)'),
    ("decision:bundle_select", "bundle_select (Scroll Boxes)"),
    ("card_select:remove", "card_select context remove"),
    ("card_select:upgrade", "card_select context upgrade"),
    ("card_select:transform", "card_select context transform"),
    ("card_select:enchant", "card_select context enchant"),
    ("card_select:grant", "card_select context grant"),
    ("potion_used", "use_potion accepted by the engine"),
    ("seek_fork", "an action after a seek forked the tree"),
    ("route:/api/live/map", "GET /api/live/map probe"),
    ("route:/api/live/timeline", "GET /api/live/timeline probe"),
    ("route:/api/live/seek", "POST /api/live/seek probe"),
    ("route:/api/live/save", "POST /api/live/save probe"),
    ("route:/api/saves", "GET /api/saves probe"),
    ("route:/api/saves/load", "POST /api/saves/load probe"),
    # M3 run mode. Same fixed-step convention.
    ("route:/api/live/start-faithful", "POST /api/live/start-faithful (run-mode scenarios)"),
    ("route:/api/live/run-mode", "POST /api/live/run-mode probe"),
    ("route:/api/live/run-mode-step", "POST /api/live/run-mode-step probe"),
    ("route:/api/live/autobattle/advance", "POST /api/live/autobattle/advance probe"),
    ("route:/api/live/autobattle/step", "POST /api/live/autobattle/step probe"),
    ("route:/api/live/autobattle/verdict", "GET /api/live/autobattle/verdict probe"),
    ("route:/api/live/ppo-policy", "POST /api/live/ppo-policy probe"),
    ("route:/api/live/wpa-review", "POST /api/live/wpa-review probe"),
    ("route:/api/live/wpa-verify", "POST /api/live/wpa-verify probe"),
    ("ppo_net", "a scenario served the committed PPO champion (values recorded)"),
    # M4: the deck routes live, then the replay viewer over the quit log.
    ("route:/api/live/deck-analysis", "GET /api/live/deck-analysis probe"),
    ("route:/api/deck/reward-preview", "POST /api/deck/reward-preview probe"),
    ("route:/api/live/quit", "POST /api/live/quit (the replay block)"),
    ("route:/api/runs", "GET /api/runs"),
    ("route:/api/load", "POST /api/load"),
    ("route:/api/timeline", "GET /api/timeline"),
    ("route:/api/map", "GET /api/map"),
    ("route:/api/step", "GET /api/step/{idx} (+ /deck-analysis)"),
    ("route:/api/deck/trend", "GET /api/deck/trend"),
    ("route:/api/wpa-review", "POST /api/wpa-review"),
    ("route:/api/recommend", "POST /api/recommend"),
    ("route:/api/search_tree", "POST /api/search_tree"),
    ("route:/api/regret-review", "POST /api/regret-review"),
    ("route:/api/mcts-runs", "GET /api/mcts-runs (+ /aggregate)"),
    ("route:/api/live/from_replay", "POST /api/live/from_replay"),
    # M2 advisors. Issued at fixed steps whatever the decision, so the tag is
    # always hit; the 200s land wherever the schedule meets a combat.
    ("route:/api/live/hint", "POST /api/live/hint probe"),
    ("route:/api/live/search_tree", "POST /api/live/search_tree probe"),
    ("route:/api/live/choice-ranker", "POST /api/live/choice-ranker probe"),
    ("route:/api/live/auto-action", "POST /api/live/auto-action probe"),
    ("route:/api/live/last-move-regret", "POST /api/live/last-move-regret probe"),
    ("route:/api/live/regret-review", "POST /api/live/regret-review probe"),
)


# ── normalization + hashing ─────────────────────────────────────────────────


def normalize(obj: Any) -> Any:
    """Return the normalized form of a decoded JSON value (see module doc)."""
    if isinstance(obj, dict):
        return {
            k: normalize(v)
            for k, v in sorted(obj.items(), key=lambda kv: kv[0])
            if k not in DROP_KEYS
        }
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, FLOAT_DECIMALS)
    return obj


def canonical_json(normalized: Any) -> str:
    """The hashed encoding: compact, sorted keys, raw non-ASCII."""
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_of(normalized: Any) -> str:
    """SHA-256 hex digest of :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()


def pretty_json(normalized: Any) -> str:
    """The stored ``responses/<i>.json`` encoding."""
    return json.dumps(normalized, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


# ── server construction ─────────────────────────────────────────────────────


class _FixedDatetime(datetime):
    """``datetime`` whose ``now()`` is pinned (patched into ``workbench.saves``)."""

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC if tz is None else tz)


#: The real champion loaders, kept so a ``"ppo": true`` scenario can restore
#: them after a stubbed one ran in the same worker process.
_REAL_LOADERS: dict[str, Any] = {}


def apply_determinism_patches(*, ppo: bool = False) -> dict[str, Any]:
    """Pin the clock and (unless ``ppo``) stub the PPO champion loader;
    return what was patched."""
    from workbench import ppo_policy, saves, server

    # Read by ``create_app`` at call time (whether to mount local ``/assets``)
    # and by ``workbench.server`` at import (the module global patched below).
    os.environ["STS2_ASSETS_BASE_URL"] = ASSETS_BASE_URL
    os.environ.pop("STS2_WB_PPO_ONNX", None)
    os.environ.pop("STS2_WB_PPO_CKPT", None)
    saves.datetime = _FixedDatetime  # ty: ignore[invalid-assignment]
    # The run log a quit writes (`game_log.GameLogger`) stamps its filename
    # and every record from the same clock; pinning it makes the log's name —
    # which `/api/mcts-runs` reports as `stem` / `path` — the same on both
    # hosts (the port stamps from the dump's fixed instant when the host
    # sends no `now`).
    import game_log

    game_log.datetime = _FixedDatetime  # ty: ignore[invalid-assignment]
    server._ASSETS_BASE_URL = ASSETS_BASE_URL
    _REAL_LOADERS.setdefault("_get_rust_agent", ppo_policy._get_rust_agent)
    _REAL_LOADERS.setdefault("_get_policy", ppo_policy._get_policy)
    ppo_policy._reset_for_tests()
    if ppo:
        ppo_policy._get_rust_agent = _REAL_LOADERS["_get_rust_agent"]
        ppo_policy._get_policy = _REAL_LOADERS["_get_policy"]
    else:

        def _no_net(character: str = "Ironclad") -> Any:
            # The browser port's wording before its host loads a champion
            # (`sts2_workbench::ppo::missing_net_error`); both hosts answer
            # the same `error` on the panel.
            key = ppo_policy._norm_char(character)
            ppo_policy._rust_errors[key] = f"no PPO net loaded for {character}"
            ppo_policy._policy_errors[key] = f"no PPO net loaded for {character}"
            return None

        ppo_policy._get_rust_agent = _no_net  # ty: ignore[invalid-assignment]
        ppo_policy._get_policy = _no_net  # ty: ignore[invalid-assignment]
    return {
        "assets_base_url": ASSETS_BASE_URL,
        "saved_at": FIXED_SAVED_AT,
        "ppo_disabled": not ppo,
    }


def build_app(work: Path):
    """Create the workbench app rooted in *work* (a fresh temp dir).

    Layout: ``<work>/site/public/images/creature_scales.json`` (so the server's
    ``images_root.parents[2] / "data"`` resolves to ``<work>/data``, a symlink
    to the sibling ``sts2-wiki/data``) and ``<work>/logs`` (so saves land in
    ``<work>/saves/workbench``). Art existence comes from the Rust crate's
    ``asset_manifest.json`` — the same bundle the port embeds.
    """
    from workbench.server import create_app

    scales = _RS_WORKBENCH_DATA / "creature_scales.json"
    manifest = _RS_WORKBENCH_DATA / "asset_manifest.json"
    for p in (scales, manifest):
        if not p.is_file():
            raise FileNotFoundError(f"missing data bundle file: {p} (run `just workbench-data`)")
    if not _WIKI_DATA.is_dir():
        raise FileNotFoundError(f"missing sibling wiki data dir: {_WIKI_DATA}")
    images = work / "site" / "public" / "images"
    images.mkdir(parents=True, exist_ok=True)
    shutil.copy(scales, images / "creature_scales.json")
    link = work / "data"
    if not link.exists():
        try:
            link.symlink_to(_WIKI_DATA, target_is_directory=True)
        except OSError:
            # Windows refuses symlinks without Developer Mode or admin rights
            # (WinError 1314); a directory junction needs neither.
            if os.name != "nt":
                raise
            import _winapi

            _winapi.CreateJunction(str(_WIKI_DATA), str(link))
    logs = work / "logs"
    logs.mkdir(exist_ok=True)
    return create_app(images_root=images, logs_dir=logs, assets_manifest=manifest)


def engine_info() -> dict[str, Any]:
    """``sts2_rs`` build identity (whatever the wheel exposes)."""
    import sts2_rs

    info: dict[str, Any] = {}
    for attr in ("version", "src_hash"):
        fn = getattr(sts2_rs, attr, None)
        if callable(fn):
            try:
                info[attr] = fn()
            except Exception:  # pragma: no cover - defensive
                info[attr] = None
    return info


# ── the scripted policy ─────────────────────────────────────────────────────


def _first_enabled(options: list[dict[str, Any]]) -> dict[str, Any] | None:
    for o in options:
        if o.get("enabled", True):
            return o
    return None


class Policy:
    """The deterministic scripted player.

    Reads only the previous response's Step (``decision`` + ``state``) plus its
    own small bookkeeping (which shop stages were done at a floor, whether a
    potion was already tried in the current combat). Never consults a hint or
    advisor route.
    """

    def __init__(self, *, neow_option: int = 0) -> None:
        self.neow_option = neow_option
        self._shop_done: dict[Any, set[str]] = {}
        self._potion_tried = False

    def choose(self, step: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        """``(action, args)`` for *step*, or ``None`` at ``game_over``."""
        decision = step.get("decision")
        state = step.get("state") or {}
        options = state.get("options") or []
        if decision == "game_over":
            return None
        if decision != "combat_play":
            self._potion_tried = False
        handler = getattr(self, f"_on_{decision}", None)
        if handler is not None:
            return handler(state, options)
        return self._fallback(options)

    @staticmethod
    def _fallback(options: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        opt = _first_enabled(options) or (options[0] if options else None)
        if opt is None:
            return "proceed", {}
        return opt["action"], dict(opt.get("args") or {})

    def _on_combat_play(self, state: dict[str, Any], _o: list) -> tuple[str, dict[str, Any]]:
        enemies = state.get("enemies") or []
        target = enemies[0].get("index") if enemies else None
        potions = [p for p in (state.get("player") or {}).get("potions") or [] if p]
        if potions and state.get("round") == 1 and not self._potion_tried:
            self._potion_tried = True
            args: dict[str, Any] = {"potion_index": 0}
            if target is not None:
                args["target_index"] = target
            return "use_potion", args
        for card in state.get("hand") or []:
            if card.get("can_play"):
                args = {"card_index": card["index"]}
                if card.get("target_type") == "AnyEnemy" and target is not None:
                    args["target_index"] = target
                return "play_card", args
        return "end_turn", {}

    def _on_map_select(self, _s: dict[str, Any], options: list) -> tuple[str, dict[str, Any]]:
        for kind in MAP_PREFERENCE:
            for opt in options:
                if str(opt.get("label", "")).startswith(kind):
                    return opt["action"], dict(opt["args"])
        return self._fallback(options)

    def _on_shop(self, state: dict[str, Any], options: list) -> tuple[str, dict[str, Any]]:
        done = self._shop_done.setdefault(state.get("floor"), set())
        by_action: dict[str, dict[str, Any]] = {}
        for opt in options:
            by_action.setdefault(opt["action"], opt)
        for stage, action in (
            ("card", "buy_card"),
            ("relic", "buy_relic"),
            ("potion", "buy_potion"),
        ):
            if stage in done:
                continue
            done.add(stage)
            if action in by_action:
                return action, dict(by_action[action]["args"])
        if "remove" not in done:
            done.add("remove")
            if state.get("card_removal_cost"):
                return "remove_card", {}
        return "leave_room", {}

    def _on_rest_site(self, state: dict[str, Any], options: list) -> tuple[str, dict[str, Any]]:
        del state  # the base policy picks off the option list alone
        # ``state.options`` is the NORMALIZED list (label == the engine's
        # ``option_id``, e.g. "SMITH" / "REST"); the raw rows are gone.
        for opt in options:
            if opt.get("label") == "SMITH" and opt.get("enabled", True):
                return opt["action"], dict(opt["args"])
        return self._fallback(options)

    def _on_event_choice(self, state: dict[str, Any], options: list) -> tuple[str, dict[str, Any]]:
        name = str(state.get("event_name") or "")
        want = self.neow_option if name.upper().startswith("NEOW") else 0
        if 0 <= want < len(options) and options[want].get("enabled", True):
            opt = options[want]
            return opt["action"], dict(opt["args"])
        return self._fallback(options)

    def _on_card_reward(self, state: dict[str, Any], options: list) -> tuple[str, dict[str, Any]]:
        cards = state.get("cards") or []
        if cards:
            return "select_card_reward", {"card_index": cards[0]["index"]}
        return self._fallback(options)

    def _on_bundle_select(self, state: dict[str, Any], options: list) -> tuple[str, dict[str, Any]]:
        bundles = state.get("bundles") or []
        if bundles:
            return "select_bundle", {"bundle_index": bundles[0]["index"]}
        return self._fallback(options)

    def _on_card_select(self, state: dict[str, Any], options: list) -> tuple[str, dict[str, Any]]:
        cards = state.get("cards") or []
        if not cards:
            return self._fallback(options)
        n = max(1, int(state.get("min_select") or 1))
        picks = [str(c["index"]) for c in cards[:n]]
        return "select_cards", {"indices": ",".join(picks)}


def _incoming_damage(enemies: list[dict[str, Any]]) -> int:
    """Total telegraphed attack damage this turn (``hits`` defaults to 1)."""
    total = 0
    for enemy in enemies:
        for intent in enemy.get("intents") or []:
            dmg = intent.get("damage")
            if isinstance(dmg, int) and dmg > 0:
                hits = intent.get("hits")
                total += dmg * (hits if isinstance(hits, int) and hits > 0 else 1)
    return total


class SurvivorPolicy(Policy):
    """The ``"survivor"`` variant used by the long scenarios.

    Same handlers as :class:`Policy` except that it tries to stay alive long
    enough to meet the act boss: greedy combat (finish a killable enemy, block
    when the telegraphed damage exceeds the current block, powers first, then
    the hardest-hitting attack on the weakest enemy), ``HEAL`` at a rest site
    when below 60% HP (``SMITH`` otherwise), and a map preference that keeps
    Elites for last. Still deterministic and still reads only the previous
    response.
    """

    MAP_PREFERENCE = (
        "RestSite",
        "Shop",
        "Treasure",
        "Monster",
        "Event",
        "Ancient",
        "Elite",
        "Boss",
    )
    HEAL_BELOW = 0.6

    def _on_map_select(self, _s: dict[str, Any], options: list) -> tuple[str, dict[str, Any]]:
        for kind in self.MAP_PREFERENCE:
            for opt in options:
                if str(opt.get("label", "")).startswith(kind):
                    return opt["action"], dict(opt["args"])
        return self._fallback(options)

    def _on_rest_site(self, state: dict[str, Any], options: list) -> tuple[str, dict[str, Any]]:
        player = state.get("player") or {}
        hp, max_hp = player.get("hp"), player.get("max_hp")
        low = (
            isinstance(hp, int)
            and isinstance(max_hp, int)
            and max_hp > 0
            and hp < self.HEAL_BELOW * max_hp
        )
        order = ("HEAL", "SMITH") if low else ("SMITH", "HEAL")
        for label in order:
            for opt in options:
                if opt.get("label") == label and opt.get("enabled", True):
                    return opt["action"], dict(opt["args"])
        return self._fallback(options)

    def _on_combat_play(self, state: dict[str, Any], _o: list) -> tuple[str, dict[str, Any]]:
        enemies = [
            e for e in state.get("enemies") or [] if isinstance(e.get("hp"), int) and e["hp"] > 0
        ]
        if not enemies:
            enemies = list(state.get("enemies") or [])
        weakest = min(
            enemies, key=lambda e: (e.get("hp") or 0) + (e.get("block") or 0), default=None
        )
        player = state.get("player") or {}
        potions = [p for p in player.get("potions") or [] if p]
        if potions and state.get("round") == 1 and not self._potion_tried:
            self._potion_tried = True
            args: dict[str, Any] = {"potion_index": 0}
            if weakest is not None:
                args["target_index"] = weakest.get("index")
            return "use_potion", args
        hand = [c for c in state.get("hand") or [] if c.get("can_play")]
        if not hand:
            return "end_turn", {}

        def play(card: dict[str, Any], target: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
            args = {"card_index": card["index"]}
            if card.get("target_type") == "AnyEnemy" and target is not None:
                args["target_index"] = target.get("index")
            return "play_card", args

        def damage(card: dict[str, Any]) -> int:
            d = (card.get("stats") or {}).get("damage")
            return d if isinstance(d, int) else 0

        def block(card: dict[str, Any]) -> int:
            b = (card.get("stats") or {}).get("block")
            return b if isinstance(b, int) else 0

        attacks = sorted(
            (c for c in hand if c.get("type") == "Attack"), key=lambda c: (-damage(c), c["index"])
        )
        # 1. finish a killable enemy
        for card in attacks:
            for enemy in sorted(enemies, key=lambda e: (e.get("hp") or 0) + (e.get("block") or 0)):
                if damage(card) >= (enemy.get("hp") or 0) + (enemy.get("block") or 0) > 0:
                    return play(card, enemy)
        # 2. block when the telegraphed damage beats the current block
        threat = _incoming_damage(enemies) - (player.get("block") or 0)
        if threat > 0:
            blockers = sorted(
                (c for c in hand if block(c) > 0), key=lambda c: (-block(c), c["index"])
            )
            if blockers:
                return play(blockers[0], weakest)
        # 3. powers, 4. hardest attack on the weakest enemy, 5. anything else
        for card in hand:
            if card.get("type") == "Power":
                return play(card, weakest)
        if attacks:
            return play(attacks[0], weakest)
        return play(hand[0], weakest)


POLICIES: dict[str, type[Policy]] = {"basic": Policy, "survivor": SurvivorPolicy}


# ── coverage ────────────────────────────────────────────────────────────────


def coverage_tags(
    prev: dict[str, Any] | None, action: str | None, step: dict[str, Any], error: Any
) -> set[str]:
    """Tags describing the Step *step* reached from *prev* via *action*."""
    tags: set[str] = set()
    decision = str(step.get("decision"))
    state = step.get("state") or {}
    tags.add(f"decision:{decision}")
    if decision == "card_select":
        ctx = str(state.get("context") or "none")
        tags.add(f"card_select:{ctx}")
        if ctx in ("remove", "upgrade"):
            tags.add("card_select:remove|upgrade")
        if ctx not in DECK_CONTEXTS and "enemies" in state:
            tags.add("combat_card_select")
        if ctx == "upgrade" and prev is not None and prev.get("decision") == "rest_site":
            tags.add("smith")
    if decision == "event_choice":
        name = state.get("event_name")
        if name == "Treasure":
            tags.add("treasure")
        # ``state.options`` is the normalized list; the enriched rows live in
        # ``event_options`` (relic offers carry relic-atlas art). A nameless
        # screen offering 3+ relics is the post-boss pick (Neow is named).
        relics = [
            o
            for o in state.get("event_options") or []
            if "relic_atlas" in str(o.get("image") or "")
        ]
        if not name and len(relics) >= 3:
            tags.add("boss_relic")
    if "room_reveal" in state:
        tags.add("room_reveal")
    if action == "use_potion" and not error:
        tags.add("potion_used")
    return tags


def print_coverage(seen: set[str], where: dict[str, str]) -> bool:
    """Print the coverage table; return True when every MUST tag was hit."""
    ok = True
    print()
    print(f"{'status':8} {'tag':32} {'first seen':38} description")
    print("-" * 110)
    for tag, desc in MUST:
        hit = tag in seen
        ok &= hit
        print(f"{'MUST ok' if hit else 'MISSING':8} {tag:32} {where.get(tag, '-'):38} {desc}")
    for tag, desc in SHOULD:
        hit = tag in seen
        print(f"{'ok' if hit else 'absent':8} {tag:32} {where.get(tag, '-'):38} {desc}")
    print()
    return ok


# ── recording one scenario ──────────────────────────────────────────────────


class Recorder:
    """Issues requests through a ``TestClient`` and records them."""

    def __init__(
        self,
        client: Any,
        out_dir: Path,
        *,
        full_steps: int,
        full: bool,
        work: Path | None = None,
    ) -> None:
        #: The scenario's scratch dir. It leaks into the fixtures through the
        #: quit log's absolute path (`/api/runs`, `/api/runs/stat?path=`,
        #: `/api/load`) and `mkdtemp` salts it, so every dump would differ;
        #: it is written as ``<work>`` instead (see `_redact`).
        self.work = str(work) if work is not None else None
        self.client = client
        self.out_dir = out_dir
        self.full_steps = full_steps if full else 0
        self.step = 0
        self.lines: list[dict[str, Any]] = []
        self.seen_pairs: set[tuple[str, str]] = set()
        self.tags: set[str] = set()
        self.where: dict[str, str] = {}
        self.full_count = 0
        (out_dir / "responses").mkdir(parents=True, exist_ok=True)
        self._jsonl = (out_dir / "requests.jsonl").open("w", encoding="utf-8")

    def close(self) -> None:
        self._jsonl.close()

    @staticmethod
    def _step_of(route: str, body: Any) -> dict[str, Any] | None:
        if not isinstance(body, dict):
            return None
        if route in (
            "/api/live/action",
            "/api/live/seek",
            "/api/live/run-mode-step",
            "/api/live/faithful-step",
            "/api/live/play-to-end",
            "/api/live/autobattle/advance",
            "/api/live/autobattle/step",
        ):
            step = body.get("step")
            return step if isinstance(step, dict) else None
        if route in (
            "/api/live/start",
            "/api/live/start-faithful",
            "/api/live/state",
            "/api/saves/load",
            "/api/live/from_replay",
        ):
            return body if "decision" in body else None
        return None

    def note(self, tags: Iterable[str], label: str) -> None:
        for tag in tags:
            if tag not in self.tags:
                self.tags.add(tag)
                self.where.setdefault(tag, label)

    def _redact(self, obj: Any) -> Any:
        """``obj`` with every occurrence of the scratch dir written as ``<work>``."""
        if self.work is None:
            return obj
        if isinstance(obj, str):
            return obj.replace(self.work, WORK_PLACEHOLDER)
        if isinstance(obj, dict):
            return {k: self._redact(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._redact(v) for v in obj]
        return obj

    def request(
        self, method: str, route: str, body: dict[str, Any] | None, *, probe: bool
    ) -> tuple[int, Any]:
        """Send one request; record it; return ``(status, decoded body)``."""
        i = len(self.lines)
        if method == "GET":
            resp = self.client.get(route)
        else:
            resp = self.client.post(route, json=body if body is not None else {})
        try:
            payload = resp.json()
        except ValueError:
            payload = {"_non_json_body": resp.text}
        norm = self._redact(normalize(payload))
        digest = sha256_of(norm)
        step = self._step_of(route, payload)
        decision = step.get("decision") if step else None
        pair = None
        if step is not None:
            ctx = str((step.get("state") or {}).get("context") or "")
            pair = (str(decision), ctx)
        new_pair = pair is not None and pair not in self.seen_pairs
        if pair is not None:
            self.seen_pairs.add(pair)
        store = probe or new_pair or resp.status_code != 200 or self.step < self.full_steps
        line = {
            "i": i,
            "method": method,
            "route": self._redact(route),
            "body": self._redact(body),
            "status": resp.status_code,
            "sha256": digest,
            "decision": decision,
        }
        self.lines.append(line)
        self._jsonl.write(json.dumps(line, ensure_ascii=False) + "\n")
        if store:
            (self.out_dir / "responses" / f"{i}.json").write_text(
                pretty_json(norm), encoding="utf-8"
            )
            self.full_count += 1
        path = route.split("?", 1)[0]
        tag = "route:/api/step" if path.startswith("/api/step/") else f"route:{path}"
        self.note([tag], f"{self.out_dir.name}#{i}")
        if resp.status_code != 200:
            self.note([f"status:{resp.status_code}"], f"{self.out_dir.name}#{i}")
        return resp.status_code, payload


def _probe_schedule(probes: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    sched: dict[int, list[dict[str, Any]]] = {}
    for p in probes:
        sched.setdefault(int(p["at"]), []).append(p)
    return sched


def run_scenario(
    scn: dict[str, Any], out_root: Path, *, full_steps: int = DEFAULT_FULL_STEPS
) -> dict[str, Any]:
    """Play one scenario and write its fixture directory; return its summary.

    Raises whatever the server/engine raised — a crashing scenario must not be
    skipped silently (the caller prints the traceback and fails the run).
    """
    from fastapi.testclient import TestClient

    patches = apply_determinism_patches(ppo=bool(scn.get("ppo")))
    name = scn["name"]
    out_dir = out_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    t0 = time.time()
    work = Path(tempfile.mkdtemp(prefix=f"golden-{name}-"))
    try:
        app = build_app(work)
        with TestClient(app) as client:
            rec = Recorder(
                client, out_dir, full_steps=full_steps, full=bool(scn.get("full")), work=work
            )
            policy_cls = POLICIES[str(scn.get("policy", "basic"))]
            policy = policy_cls(neow_option=int(scn.get("neow_option", 0)))
            sched = _probe_schedule(scn.get("probes") or [])
            max_steps = int(scn["max_steps"])
            if scn.get("start") == "faithful":
                # A run-mode run (integer seed, the PPO env's constructor) at
                # a tiny combat budget so the agent probes stay quick.
                start_route = "/api/live/start-faithful"
                start_body = {
                    "character": scn["character"],
                    "seed": scn["seed"],
                    "ascension": int(scn["ascension"]),
                    "sims": int(scn.get("combat_sims", 8)),
                    "boss_sims": int(scn.get("combat_boss_sims", 16)),
                    "threads": 1,
                }
            else:
                start_route = "/api/live/start"
                start_body = {
                    "character": scn["character"],
                    "seed": scn["seed"],
                    "ascension": int(scn["ascension"]),
                    "act1": scn["act1"],
                    "backend": "rust",
                }
            status, body = rec.request("POST", start_route, start_body, probe=False)
            if status != 200:
                raise RuntimeError(f"{name}: {start_route} -> {status}: {body}")
            current: dict[str, Any] = body
            rec.note(coverage_tags(None, None, current, None), f"{name}#0")
            if scn.get("ppo"):
                rec.note(["ppo_net"], f"{name}#0")
            prev: dict[str, Any] | None = None
            after_seek = False
            ended = False
            while rec.step < max_steps:
                for probe in sched.get(rec.step, []):
                    current, after_seek = _run_probe(rec, probe, current) or (current, after_seek)
                choice = policy.choose(current)
                if choice is None:
                    ended = True
                    break
                action, args = choice
                status, body = rec.request(
                    "POST", "/api/live/action", {"action": action, "args": args}, probe=False
                )
                rec.step += 1
                if status != 200 or not isinstance(body, dict) or "step" not in body:
                    raise RuntimeError(
                        f"{name}: step {rec.step} {action} {args} -> {status}: {body}"
                    )
                error = body.get("error")
                prev, current = current, body["step"]
                tags = coverage_tags(prev, action, current, error)
                if after_seek and not error:
                    tags.add("seek_fork")
                    after_seek = False
                rec.note(tags, f"{name}#{len(rec.lines) - 1}")
            if not ended and str(current.get("decision")) == "game_over":
                ended = True
            if scn.get("replay", True):
                _replay_block(rec, full=bool(scn.get("full")))
        rec.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    elapsed = time.time() - t0
    meta = {
        "scenario": scn,
        "engine": engine_info(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "patches": patches,
        "full_steps": rec.full_steps,
        "steps_played": rec.step,
        "requests": len(rec.lines),
        "full_responses": rec.full_count,
        "reached_game_over": bool(ended and str(current.get("decision")) == "game_over"),
        "final_floor": (current.get("state") or {}).get("floor"),
        "final_act": (current.get("state") or {}).get("act"),
        "coverage": sorted(rec.tags),
    }
    (out_dir / "meta.json").write_text(pretty_json(meta), encoding="utf-8")
    return {
        "name": name,
        "seconds": elapsed,
        "steps": rec.step,
        "requests": len(rec.lines),
        "full_responses": rec.full_count,
        "game_over": meta["reached_game_over"],
        "floor": meta["final_floor"],
        "act": meta["final_act"],
        "tags": sorted(rec.tags),
        "where": rec.where,
    }


def _replay_block(rec: Recorder, *, full: bool) -> None:
    """End the run and drive the replay viewer over the log it left: list,
    load, the projections, the deck routes, the reviews, the searches, the
    MCTS Runs page, then "play from here" — the exact sequence the SPA's
    "Review replay" hand-off issues, at the advisor probes' tiny budgets.

    The log's filename is the pinned clock + character + seed on both hosts,
    and the Rust replay stores the log under it; `/api/load`'s `path` is this
    host's absolute path, which the port's test maps by basename.

    The bulky bodies (the whole timeline, the map graphs, every step, the
    reviews, the resumed run) are stored only for a ``full`` scenario — the
    hash is the gate; the three long scenarios keep the diffable copies.
    """
    status, body = rec.request("POST", "/api/live/quit", None, probe=True)
    if status != 200 or not isinstance(body, dict) or not body.get("log_path"):
        return
    log_path = str(body["log_path"])
    # `/api/runs/stat` reports the log's mtime: pin it to the dump's clock.
    os.utime(log_path, (PINNED_EPOCH, PINNED_EPOCH))
    rec.request("GET", "/api/runs", None, probe=True)
    rec.request("GET", f"/api/runs/stat?path={log_path}", None, probe=True)
    status, loaded = rec.request("POST", "/api/load", {"path": log_path}, probe=True)
    if status != 200 or not isinstance(loaded, dict):
        return
    count = int(loaded.get("count") or 0)
    rec.request("GET", "/api/timeline", None, probe=full)
    rec.request("GET", "/api/map", None, probe=full)
    # A few steps: the first, a middle one, the last, and the first fight.
    status, tl = rec.request("GET", "/api/timeline", None, probe=full)
    steps = (tl or {}).get("steps") or [] if isinstance(tl, dict) else []
    combat_idx = next((s["idx"] for s in steps if s.get("decision") == "combat_play"), None)
    picks = sorted(
        {0, count // 2, max(0, count - 1)} | ({combat_idx} if combat_idx is not None else set())
    )
    for k in picks:
        rec.request("GET", f"/api/step/{k}", None, probe=full)
    rec.request("GET", f"/api/step/{picks[-1]}/deck-analysis", None, probe=True)
    rec.request("GET", f"/api/step/{count + 5}", None, probe=True)
    rec.request("GET", "/api/deck/trend", None, probe=True)
    rec.request("POST", "/api/wpa-review", {"config": {"top_k": 3}}, probe=True)
    if combat_idx is not None:
        rec.request(
            "POST", "/api/recommend", {"idx": combat_idx, "config": ADVISOR_SEARCH}, probe=True
        )
        rec.request(
            "POST", "/api/search_tree", {"idx": combat_idx, "config": ADVISOR_SEARCH}, probe=full
        )
    rec.request("POST", "/api/search_tree", {"idx": 0, "config": ADVISOR_SEARCH}, probe=True)
    rec.request(
        "POST", "/api/regret-review", {"config": {**ADVISOR_REGRET, "top_n": 2}}, probe=full
    )
    rec.request("GET", "/api/mcts-runs", None, probe=True)
    rec.request("GET", "/api/mcts-runs/aggregate", None, probe=True)
    resume_idx = combat_idx if combat_idx is not None else min(3, max(0, count - 2))
    rec.request("POST", "/api/live/from_replay", {"idx": resume_idx}, probe=True)
    rec.request("GET", "/api/live/timeline", None, probe=full)
    rec.request("GET", "/api/live/state", None, probe=full)


def _run_probe(
    rec: Recorder, probe: dict[str, Any], current: dict[str, Any]
) -> tuple[dict[str, Any], bool] | None:
    """Issue one probe; return the new ``(current step, after_seek)`` or None."""
    kind = probe["kind"]
    if kind == "state":
        status, body = rec.request("GET", "/api/live/state", None, probe=True)
        return (body, False) if status == 200 and isinstance(body, dict) else None
    if kind == "map":
        rec.request("GET", "/api/live/map", None, probe=True)
        return None
    if kind == "timeline":
        rec.request("GET", "/api/live/timeline", None, probe=True)
        return None
    if kind == "seek":
        status, tl = rec.request("GET", "/api/live/timeline", None, probe=True)
        if status != 200 or not isinstance(tl, dict):
            return None
        path = list(tl.get("active_path") or [])
        if not path:
            return None
        back = int(probe.get("back", 5))
        node_id = path[max(0, len(path) - 1 - back)]
        status, body = rec.request("POST", "/api/live/seek", {"node_id": node_id}, probe=True)
        if status == 200 and isinstance(body, dict) and isinstance(body.get("step"), dict):
            return body["step"], True
        return None
    if kind == "save":
        rec.request("POST", "/api/live/save", {"name": probe.get("name", "golden mid")}, probe=True)
        return None
    if kind == "saves_list":
        rec.request("GET", "/api/saves", None, probe=True)
        return None
    if kind == "load":
        status, body = rec.request(
            "POST", "/api/saves/load", {"name": probe.get("name", "golden mid")}, probe=True
        )
        if status == 200 and isinstance(body, dict) and "decision" in body:
            return body, False
        return None
    if kind in _ADVISOR_PROBES:
        route, body = _ADVISOR_PROBES[kind]
        rec.request("POST", route, body, probe=True)
        return None
    # M3 run mode. `run_mode` / `autobattle_verdict` / `ppo_policy` /
    # `wpa_review` are read-only; the stepping kinds ADVANCE the run and hand
    # the new step back so the scripted policy continues from it.
    if kind == "run_mode":
        rec.request("POST", "/api/live/run-mode", dict(probe.get("config") or {}), probe=True)
        return None
    if kind == "deck_analysis":
        rec.request("GET", "/api/live/deck-analysis", None, probe=True)
        return None
    if kind == "reward_preview":
        # The current deck and, at a card reward, the offered cards (else a
        # fixed pair) — the paired-seed deltas are what the probe holds.
        state = current.get("state") or {}
        deck = (state.get("player") or {}).get("deck") or []
        cands = state.get("cards") if current.get("decision") == "card_reward" else None
        if not cands:
            cands = [{"index": 0, "id": "INFLAME"}, {"index": 1, "id": "CLEAVE", "upgraded": True}]
        body = {
            "deck": [{"id": c.get("id"), "upgraded": bool(c.get("upgraded"))} for c in deck],
            "candidates": [
                {"index": c.get("index", i), "id": c.get("id"), "upgraded": bool(c.get("upgraded"))}
                for i, c in enumerate(cands)
            ],
        }
        rec.request("POST", "/api/deck/reward-preview", body, probe=True)
        return None
    if kind in ("run_mode_step", "autobattle_advance", "autobattle_step"):
        route = {
            "run_mode_step": "/api/live/run-mode-step",
            "autobattle_advance": "/api/live/autobattle/advance",
            "autobattle_step": "/api/live/autobattle/step",
        }[kind]
        status, body = rec.request("POST", route, None, probe=True)
        if status == 200 and isinstance(body, dict) and isinstance(body.get("step"), dict):
            return body["step"], False
        return None
    if kind == "autobattle_verdict":
        rec.request("GET", "/api/live/autobattle/verdict", None, probe=True)
        return None
    if kind == "ppo_policy":
        rec.request("POST", "/api/live/ppo-policy", None, probe=True)
        return None
    if kind == "wpa_review":
        rec.request("POST", "/api/live/wpa-review", {"config": {"top_k": 3}}, probe=True)
        return None
    if kind == "wpa_verify":
        # The review names the turning point; the verify plays two short
        # agent arms from it and restores the active leaf, so the scripted
        # policy continues where it was (the arms stay in the tree).
        status, review = rec.request(
            "POST", "/api/live/wpa-review", {"config": {"top_k": 1}}, probe=True
        )
        tps = (review or {}).get("turning_points") if isinstance(review, dict) else None
        if status == 200 and tps:
            node_id = tps[0].get("node_id")
            if node_id is not None:
                rec.request(
                    "POST",
                    "/api/live/wpa-verify",
                    {
                        "node_id": int(node_id),
                        "max_steps": int(probe.get("max_steps", 12)),
                        "max_seconds": 3600.0,
                    },
                    probe=True,
                )
        return None
    raise ValueError(f"unknown probe kind: {kind!r}")


#: The search config every advisor probe carries: the deployed honest triple
#: at a budget small enough to keep the dump quick, seeded so the search is
#: reproducible. The Rust port clamps ``threads`` to 1 (no thread pool in
#: wasm), so the probe pins 1 too — with ``threads > 1`` Python would run the
#: root-parallel search and the hashes could never match.
ADVISOR_SEARCH = {
    "sims": 48,
    "seed": 1,
    "threads": 1,
    "c_puct": 2.5,
    "depth": 200,
    "rollout_policy": "heuristic",
    "hidden_info": "beliefdraw",
    "draw_widening_c": 2.0,
}
#: The regret routes re-score every combat position on the line; cheaper.
ADVISOR_REGRET = {**ADVISOR_SEARCH, "sims": 24, "seed": 2}
#: probe kind -> (route, body). The advisor probes never mutate the run.
_ADVISOR_PROBES: dict[str, tuple[str, dict[str, Any] | None]] = {
    "hint": ("/api/live/hint", {"config": ADVISOR_SEARCH}),
    "search_tree": ("/api/live/search_tree", {"config": ADVISOR_SEARCH}),
    "choice_ranker": ("/api/live/choice-ranker", {"config": ADVISOR_SEARCH}),
    "auto_action": ("/api/live/auto-action", None),
    "last_move_regret": ("/api/live/last-move-regret", {"config": ADVISOR_REGRET}),
    "regret_review": ("/api/live/regret-review", {"config": {**ADVISOR_REGRET, "top_n": 2}}),
}


# ── scenarios ───────────────────────────────────────────────────────────────


def standard_probes(max_steps: int) -> list[dict[str, Any]]:
    """The probe set every scenario carries."""
    probes: list[dict[str, Any]] = [
        {"at": 40, "kind": "map"},
        {"at": 41, "kind": "timeline"},
        {"at": 42, "kind": "seek", "back": 5},
        {"at": 60, "kind": "save", "name": "golden mid"},
        {"at": 61, "kind": "saves_list"},
        {"at": 62, "kind": "load", "name": "golden mid"},
    ]
    probes.extend({"at": at, "kind": "state"} for at in range(25, max_steps + 1, 25))
    # M2 advisors: the search-backed routes at a few fixed steps (most land in
    # a combat somewhere along the 150-step scripts; off combat they record a
    # 409), the coach right after a move, the run review once, "play best" at
    # the state probes.
    for at in (30, 55, 80, 105, 130):
        probes.append({"at": at, "kind": "hint"})
        probes.append({"at": at, "kind": "choice_ranker"})
        probes.append({"at": at + 1, "kind": "last_move_regret"})
    # A whole 48-sim tree is ~35 KB stored; two per scenario keep the fixture
    # under the size budget while still landing in a fight on most scripts.
    for at in (55, 105):
        probes.append({"at": at, "kind": "search_tree"})
    probes.extend({"at": at, "kind": "auto_action"} for at in range(25, max_steps + 1, 25))
    probes.append({"at": 100, "kind": "regret_review"})
    # M3: the PPO panel and the WPA review answer "no net" here (the stub);
    # the labelled option list is what the probe holds to parity.
    for at in (25, 75):
        probes.append({"at": at, "kind": "ppo_policy"})
    probes.append({"at": 75, "kind": "wpa_review"})
    # M4: the deck grade of the live deck, and the reward preview's paired
    # deltas (at a card reward when the schedule lands on one).
    probes.append({"at": 50, "kind": "deck_analysis"})
    for at in (35, 85):
        probes.append({"at": at, "kind": "reward_preview"})
    return sorted(probes, key=lambda p: p["at"])  # stable: file order within a step


def run_mode_probes(max_steps: int, *, ppo: bool) -> list[dict[str, Any]]:
    """The probe set of a run-mode scenario: the standard set plus the agent
    stepping through the opening (under the ``ppo`` brain when the scenario
    serves the champion — a stubbed run keeps the heuristic brain, whose
    off-combat fallback is each host's own), an autobattle leg (skip,
    verdict, watch), the PPO panel every 10 steps, a mid-run budget change,
    the WPA review twice and one verify."""
    probes = standard_probes(max_steps)
    if ppo:
        probes.append({"at": 1, "kind": "run_mode", "config": {"off_combat_brain": "ppo"}})
    probes.extend({"at": at, "kind": "run_mode_step"} for at in (2, 3, 4, 5, 6))
    probes.append({"at": 20, "kind": "autobattle_advance"})
    probes.append({"at": 21, "kind": "autobattle_verdict"})
    probes.extend({"at": at, "kind": "autobattle_step"} for at in (40, 41, 42, 43))
    probes.append({"at": 44, "kind": "autobattle_verdict"})
    probes.extend({"at": at, "kind": "ppo_policy"} for at in range(10, max_steps + 1, 10))
    probes.append({"at": 70, "kind": "run_mode", "config": {"combat_sims": 12}})
    probes.extend({"at": at, "kind": "run_mode_step"} for at in (71, 72, 73, 74, 75, 76))
    probes.append({"at": 90, "kind": "run_mode", "config": {"off_combat_brain": "heuristic"}})
    probes.extend({"at": at, "kind": "run_mode_step"} for at in (91, 92, 93))
    if ppo:
        probes.append({"at": 94, "kind": "run_mode", "config": {"off_combat_brain": "ppo"}})
    probes.append({"at": 50, "kind": "wpa_verify", "max_steps": 12})
    probes.append({"at": 60, "kind": "wpa_review"})
    probes.append({"at": 140, "kind": "wpa_review"})
    return sorted(probes, key=lambda p: p["at"])


def scenario_matrix() -> list[dict[str, Any]]:
    """The committed scenario set: 5 chars x 2 seeds x A{0,10} x 2 acts + 3 long."""
    out: list[dict[str, Any]] = []
    for character in CHARACTERS:
        for seed_no, seed in enumerate(SEEDS):
            for ascension in (0, 10):
                for act1 in ("overgrowth", "underdocks"):
                    name = f"{character.lower()}_{seed.lower()}_a{ascension}_{act1}"
                    out.append(
                        {
                            "name": name,
                            "character": character,
                            "seed": seed,
                            "ascension": ascension,
                            "act1": act1,
                            "max_steps": 150,
                            "neow_option": seed_no,
                            "probes": standard_probes(150),
                        }
                    )
    # M3 run mode: faithful (integer-seed) runs at a tiny combat budget. Two
    # serve the committed champion (values, panel probabilities, WPA), one
    # runs without a net at A10 under the heuristic brain.
    for character, seed_no, ascension, ppo in (
        ("Ironclad", 1, 0, True),
        ("Silent", 2, 0, True),
        ("Defect", 3, 10, False),
    ):
        out.append(
            {
                "name": f"runmode_{character.lower()}_run{seed_no}_a{ascension}",
                "character": character,
                "seed": f"run_{seed_no}",
                "ascension": ascension,
                "act1": "overgrowth",
                "start": "faithful",
                "combat_sims": 8,
                "combat_boss_sims": 16,
                "ppo": ppo,
                "max_steps": 150,
                "neow_option": seed_no % 2,
                "probes": run_mode_probes(150, ppo=ppo),
            }
        )
    for character in ("Ironclad", "Silent", "Defect"):
        name = f"long_{character.lower()}_{SEEDS[0].lower()}_a0_roll"
        out.append(
            {
                "name": name,
                "character": character,
                "seed": SEEDS[0],
                "ascension": 0,
                "act1": "roll",
                "max_steps": 2000,
                "neow_option": 0,
                "policy": "survivor",
                "full": True,
                "probes": standard_probes(2000),
            }
        )
    return out


def write_scenarios(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.json"):
        old.unlink()
    scenarios = scenario_matrix()
    for scn in scenarios:
        (directory / f"{scn['name']}.json").write_text(pretty_json(scn), encoding="utf-8")
    return len(scenarios)


def load_scenarios(directory: Path, only: list[str] | None) -> list[dict[str, Any]]:
    scenarios = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))
    ]
    if only:
        scenarios = [s for s in scenarios if any(fnmatch.fnmatch(s["name"], pat) for pat in only)]
    return scenarios


# ── driver ──────────────────────────────────────────────────────────────────


def _worker(args: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
    scn, out_root, full_steps = args
    try:
        return run_scenario(scn, Path(out_root), full_steps=full_steps)
    except BaseException:  # report, never skip silently
        return {"name": scn["name"], "error": traceback.format_exc()}


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", action="append", help="scenario name glob (repeatable)")
    ap.add_argument("--full-steps", type=int, default=DEFAULT_FULL_STEPS, help="rule (d) cutoff")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument(
        "--write-scenarios", action="store_true", help="(re)generate the matrix and exit"
    )
    ap.add_argument(
        "--size-limit-mb", type=float, default=15.0, help="warn when the output exceeds this"
    )
    args = ap.parse_args(argv)

    if args.write_scenarios:
        n = write_scenarios(args.scenarios)
        print(f"wrote {n} scenarios to {args.scenarios}")
        return 0

    try:
        import sts2_rs  # noqa: F401
    except ImportError:
        print(
            "ERROR: `import sts2_rs` failed — build the wheel first: `just build-rust`",
            file=sys.stderr,
        )
        return 2

    scenarios = load_scenarios(args.scenarios, args.only)
    if not scenarios:
        print(f"no scenarios matched under {args.scenarios}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"engine: {engine_info()}  scenarios: {len(scenarios)}  jobs: {args.jobs}  out: {args.out}"
    )

    t0 = time.time()
    jobs = [(scn, str(args.out), args.full_steps) for scn in scenarios]
    results: list[dict[str, Any]] = []
    if args.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for res in pool.map(_worker, jobs):
                results.append(res)
                _print_result(res)
    else:
        for job in jobs:
            res = _worker(job)
            results.append(res)
            _print_result(res)

    seen: set[str] = set()
    where: dict[str, str] = {}
    failures = [r for r in results if "error" in r]
    for r in results:
        if "error" in r:
            continue
        for tag in r["tags"]:
            if tag not in seen:
                seen.add(tag)
                where[tag] = r["where"].get(tag, r["name"])
    ok = print_coverage(seen, where)
    size = dir_size(args.out)
    print(
        f"total: {len(results)} scenarios, {sum(r.get('requests', 0) for r in results)} requests, "
        f"{sum(r.get('full_responses', 0) for r in results)} full responses, "
        f"{size / 1e6:.1f} MB in {args.out}, {time.time() - t0:.0f}s wall"
    )
    if size > args.size_limit_mb * 1e6:
        print(
            f"WARNING: output exceeds {args.size_limit_mb:.0f} MB — re-run with a smaller "
            f"--full-steps (rule d) e.g. --full-steps 40"
        )
    if failures:
        print(f"\n{len(failures)} scenario(s) FAILED:", file=sys.stderr)
        for r in failures:
            print(f"--- {r['name']} ---\n{r['error']}", file=sys.stderr)
        return 1
    if not ok:
        print(
            "ERROR: a MUST coverage item is missing. Add a scenario (or raise max_steps / "
            "move a probe) that reaches it — see the table above.",
            file=sys.stderr,
        )
        return 1
    return 0


def _print_result(res: dict[str, Any]) -> None:
    if "error" in res:
        print(f"  {res['name']:44} FAILED")
        return
    print(
        f"  {res['name']:44} {res['steps']:5d} steps {res['requests']:5d} req "
        f"{res['full_responses']:4d} full {res['seconds']:6.1f}s  act {res['act']} floor {res['floor']}"
        f"{'  game_over' if res['game_over'] else ''}"
    )


if __name__ == "__main__":
    sys.exit(main())
