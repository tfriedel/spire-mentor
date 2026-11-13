"""Eval harness for the C# HEADLESS runner with the live AI stack (rung 3.5).

The missing rung between rung 3 (Rust engine as both world and search
model, PPO meta: 50% honest win rate) and the live game via SpireMentor
(0/12): here the **real game DLL** is the world (real content, real
mechanics via ``Sts2Headless``) while the Rust engine is only the MCTS
search model — exactly like live, but with no bridge, no UI timing, no
mod. If this rung collapses too, the remaining live gap is content /
search-model mismatch; if it holds ~50%, the gap is in the live bridge
layer specifically.

Reuses the live decision stack verbatim wherever the dialects allow:

* Non-combat: each headless decision is converted to the SpireMentor
  message shape and answered by :func:`realgame_ppo.
  pick_noncombat_action_ppo` — the same builders, net, session map
  cache, and fallbacks the live sweeps use. The bridge-dialect answer
  is then mapped back to a headless ``action`` command.
* Combat: driven by a **persistent** :class:`~combat_mirror.CombatMirror` —
  one engine state per fight, advanced by the actions the DLL actually
  accepts, with every observable re-imposed from the wire each decision.
  Searches run on a clone of it, so they see the RNG cursors and draw-pile
  pins a per-decision reconstruction throws away. The ranked tags are walked
  best-first, skipping cards the game marks ``can_play=False`` (the live
  playable-mask, PR #260).

  A mid-combat ``card_select`` is answered off that same mirror: the played
  card already parked it on a choice, the DLL now attaches the full board
  (``state["combat"]``), and the live option list is injected over the
  engine's own derivation — which is demoted to an assertion. That replaces
  the old three-tier chain (pending-choice replay → ``prompt_key`` heuristic
  → random) outright. What the chain cost, measured on the realdll_v4 sweep
  (60 runs, 635 prompts): 263 replays failed the match gate, 121 were never
  attempted, 175 were answered at random.

  Out-of-combat selections (rest-site upgrade, event grant, shop removal)
  still go to the trained PPO surface.

Outcome aggregation and reports reuse :mod:`realgame_eval` wholesale, so
``report.md`` is directly comparable with the live sweeps.

Usage::

    uv run python python/headless_eval.py --runs 20 --character Ironclad \\
        --seed-base 1000 --sims-regular 1000 --sims-boss 4000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sts2_client
import sts2_rs
from combat_mirror import (
    CombatMirror,
    MirrorDivergence,
    MirrorPolicy,
    policy_from_env,
    recognize_card_select,
    skip_arity_from_world,
)
from ppo import vocab
from realgame_eval import RunOutcome, build_report, render_markdown, write_report
from realgame_mcts import DEFAULT_SIMS_BOSS, DEFAULT_SIMS_REGULAR
from realgame_ppo import PpoPolicyCache, pick_noncombat_action_ppo
from realgame_random import JsonlLogger
from shadow_replay import diff_states
from sts2_client import Game
from sts2_hint import adapter
from sts2_hint.adapter import drop_counts, reset_drop_counts
from sts2_hint.display_maps import loc_id
from sts2_hint.recommender import (
    END_TURN_STABLE_IDX,
    POTION_STABLE_IDX_BASE,
    RecommendConfig,
    recommend_all,
    tags_from_result,
)
from workbench.backends.rust_game import RustGame

logger = logging.getLogger(__name__)


class WireRunTracer:
    """Per-floor world trace for the real-DLL side of the deck/HP-arrival diff.

    Mirrors ``play_full_run_ppo.RunTracer``'s schema (``ev: room`` with
    ``floor/hp/max_hp/gold/deck_size``, ``ev: combat`` with
    ``hp_before/hp_after/hp_lost/roster/is_elite``, ``ev: run_end``) but reads
    the DLL WIRE state instead of a native ``RunManager``, so an engine-world
    trace and a real-DLL trace of the same string-seed diff directly (deck
    quality + HP at matched floors, HP-bleed per fight). Seed-keyed file, so
    no jsonl↔seed matching guesswork.
    """

    def __init__(self, path: Path, seed: int) -> None:
        self._f = path.open("w", encoding="utf-8")
        self._seed = seed
        self._last_room_key: tuple[int, str] | None = None
        self._in_combat = False
        self._combat_rec: dict[str, Any] | None = None
        self._prev_snap: dict[str, Any] | None = None
        self._prev_in_combat = False
        self._prev_where: dict[str, Any] | None = None
        self._prev_decision: str | None = None

    def _emit(self, rec: dict[str, Any]) -> None:
        self._f.write(json.dumps({"seed": self._seed, **rec}, separators=(",", ":")) + "\n")
        self._f.flush()

    @staticmethod
    def _resource_id(entry: Any, *keys: str) -> str | None:
        """Canonical SCREAMING_SNAKE stem for one relic/potion/card entry.

        ``None`` when the entry carries none of ``keys`` — the caller drops it
        rather than bucketing it under the string ``"None"``, which would have
        merged every id-less relic into one phantom relic that appears to be
        gained and lost as the belt changes.
        """
        for key in keys:
            raw = entry.get(key) if isinstance(entry, dict) else None
            if raw:
                return loc_id(str(raw))
        return None

    @classmethod
    def _snapshot(cls, player: dict[str, Any]) -> dict[str, Any]:
        """Comparable resource snapshot: deck cards carry a ``+`` suffix when
        upgraded so an in-place smith shows up as remove(X)+add(X+).

        Every id goes through :func:`sts2_hint.display_maps.loc_id`, which is
        what makes this trace diffable against ``play_full_run_ppo``'s engine
        twin. The two worlds name the same object three different ways — DLL
        ``SILVER_CRUCIBLE`` / ``CARD.UNRELENTING`` / ``ATTACK_POTION`` vs
        engine ``RELIC.SILVER_CRUCIBLE`` / ``Unrelenting`` / ``AttackPotion``
        — so an un-normalised diff reports 100% spurious divergence on every
        relic, card and potion.

        Potions are keyed on ``entry`` (the registry id ``PlayerSummary`` emits
        for this purpose), never on ``name``: the display title is localized
        and, for the potions whose title differs from their entry words, is not
        recoverable back to an id at all.
        """
        deck = player.get("deck") or []
        relics = (cls._resource_id(r, "id") for r in (player.get("relics") or []))
        potions = (cls._resource_id(p, "entry", "id") for p in (player.get("potions") or []))
        cards = (
            (cls._resource_id(c, "id"), c.get("upgraded")) for c in deck if isinstance(c, dict)
        )
        return {
            "gold": player.get("gold"),
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "relics": Counter(r for r in relics if r is not None),
            "potions": Counter(p for p in potions if p is not None),
            "cards": Counter(
                cid + ("+" if upgraded else "") for cid, upgraded in cards if cid is not None
            ),
        }

    def _emit_delta(self, snap: dict[str, Any], in_combat: bool) -> None:
        """One ``ev: delta`` per observe whose resources changed vs the last
        observe — attributes every relic/potion/card/gold/max-hp change (and
        out-of-combat HP change) to the decision that caused it, so rest-vs-
        smith, shop buys, and event outcomes are readable from the trace.

        The record is stamped with the location of the PREVIOUS observe, to
        match ``after_decision``: the change belongs where it was caused, not
        where the driver happens to be standing one decision later. Stamping
        the current location put every event outcome at the map node the
        player advanced to (``event_choice`` records split 153 ``Event`` / 83
        ``Map`` on the same corpus) and made the two arms disagree about where
        the same pickup happened.
        """
        prev = self._prev_snap
        if prev is None:
            return
        changes: dict[str, Any] = {}
        for kind in ("relics", "potions", "cards"):
            added = list((snap[kind] - prev[kind]).elements())
            removed = list((prev[kind] - snap[kind]).elements())
            if added:
                changes[f"{kind}_added"] = added
            if removed:
                changes[f"{kind}_removed"] = removed
        for field in ("gold", "max_hp"):
            a, b = prev.get(field), snap.get(field)
            if a is not None and b is not None and a != b:
                changes[f"{field}_delta"] = b - a
        # HP outside combat only — in-combat flow is the combat record's job.
        if not in_combat and not self._prev_in_combat:
            a, b = prev.get("hp"), snap.get("hp")
            if a is not None and b is not None and a != b:
                changes["hp_delta"] = b - a
        if changes:
            self._emit(
                {
                    "ev": "delta",
                    **(self._prev_where or {}),
                    "after_decision": self._prev_decision,
                    **changes,
                }
            )

    def observe(self, state: dict[str, Any]) -> None:
        """Call once per decision, before answering it."""
        ctx = state.get("context") or {}
        player = state.get("player") or {}
        floor = ctx.get("total_floor")
        room_type = ctx.get("room_type")
        hp = player.get("hp")
        deck_size = player.get("deck_size")
        if deck_size is None:
            deck_size = len(player.get("deck") or [])
        snap = self._snapshot(player)
        if isinstance(floor, int) and room_type and (floor, room_type) != self._last_room_key:
            self._last_room_key = (floor, room_type)
            self._emit(
                {
                    "ev": "room",
                    "floor": floor,
                    "act": ctx.get("act"),
                    "room_type": room_type,
                    "hp": hp,
                    "max_hp": player.get("max_hp"),
                    "gold": player.get("gold"),
                    "deck_size": deck_size,
                    "upgrades": sum(1 for c in (player.get("deck") or []) if c.get("upgraded")),
                    "relics": sorted(snap["relics"].elements()),
                    "potions": sorted(snap["potions"].elements()),
                }
            )
        in_combat = state.get("decision") == "combat_play"
        self._emit_delta(snap, in_combat)
        self._prev_snap = snap
        self._prev_in_combat = in_combat
        self._prev_where = {"floor": floor, "act": ctx.get("act"), "room_type": room_type}
        self._prev_decision = state.get("decision")
        if in_combat and not self._in_combat:
            self._in_combat = True
            self._combat_rec = {
                "ev": "combat",
                "floor": floor,
                "act": ctx.get("act"),
                "room_type": room_type,
                "is_elite": bool(ctx.get("is_elite") or (ctx.get("room_type") == "Elite")),
                "hp_before": hp,
                "deck_size": deck_size,
            }
        elif not in_combat and self._in_combat:
            self._in_combat = False
            if self._combat_rec is not None:
                self._combat_rec["hp_after"] = hp
                self._combat_rec["hp_lost"] = (self._combat_rec.get("hp_before") or 0) - (hp or 0)
                self._emit(self._combat_rec)
                self._combat_rec = None

    def finish(self, state: dict[str, Any], won: bool) -> None:
        if self._in_combat and self._combat_rec is not None:
            player = state.get("player") or {}
            hp_after = player.get("hp")
            self._combat_rec["hp_after"] = hp_after
            # Set hp_lost here too (not just on the observe() combat-close
            # path): a run that ends mid-fight still owes the death fight's
            # bleed to diag_hp_flow / diag_deck_hp_arrival, which read
            # `hp_lost`. Omitting it counted the final DLL fight as 0 loss.
            self._combat_rec["hp_lost"] = (self._combat_rec.get("hp_before") or 0) - (hp_after or 0)
            self._emit(self._combat_rec)
            self._combat_rec = None
        player = state.get("player") or {}
        self._emit(
            {
                "ev": "run_end",
                "victory": bool(won),
                "floor": (state.get("context") or {}).get("total_floor"),
                "hp": player.get("hp"),
                "deck_size": player.get("deck_size") or len(player.get("deck") or []),
            }
        )
        self._f.close()


#: Headless decision types that are terminal for a run.
_TERMINAL_DECISIONS = frozenset({"game_over"})

#: Abort a run once the game returns a byte-identical state this many times in
#: a row — nothing we send is having any effect. The real DLL can wedge combat
#: in a non-play phase (EndTurn recovery ladder exhausted, ActionExecutor stuck
#: on a WaitUntilQueue TCS); every subsequent action then echoes the same
#: CombatPlayState at ~2s each, grinding the full ``max_steps`` cap (~100 min,
#: seeds 1007/1009 of the 2026-07-16 rung-3.5 sweep).
_STALL_REPEAT_LIMIT = 25

#: Sliding window for the cyclic-stall breaker. A wedged DLL can ALTERNATE
#: states instead of echoing one: play -> "error" state -> proceed -> same
#: combat state -> identical play -> ... (seed 1025, 2026-07-18 sweep: 826
#: identical rejected plays over 24 min; the consecutive-identical check above
#: never fires because the interleaved error state resets it). If any single
#: state repeats ``_STALL_REPEAT_LIMIT`` times within this window, the run is
#: stalled regardless of what is interleaved between the repeats.
_STALL_WINDOW = 80


# ---------------------------------------------------------------------------
# Headless -> bridge-dialect state conversion (non-combat surfaces).
# ---------------------------------------------------------------------------


def _plain_card_id(raw: Any) -> str:
    """``"CARD.STRIKE_IRONCLAD"`` / ``"STRIKE_IRONCLAD"`` -> plain entry id."""
    s = str(raw or "")
    return s.rsplit(".", 1)[-1] if "." in s else s


def _english(text: Any) -> str:
    """Bilingual ``"English / Other"`` -> English half (idempotent)."""
    s = str(text or "")
    return s.split(" / ", 1)[0].strip()


def _potion_obs_id(p: dict[str, Any]) -> str:
    """Potion entry -> the id the PPO vocab can resolve.

    The real DLL sends only a resolved display name ("Explosive Ampoule"),
    which ``ppo.vocab.canonical`` maps fine. Emulated dialects (RustGame
    csharp) leak the loc-key form ``"VULNERABLE_POTION.title"`` instead —
    ``canonical`` keeps the part AFTER the first dot, collapsing every such
    potion to the OOV token "TITLE" (2026-07-15 featurizer audit: flipped
    ~half the sampled map_select/card_reward net decisions). The loc-key stem
    IS the registry id, so strip the suffix rather than feeding it downstream.
    """
    pid = p.get("id")
    if pid:
        return str(pid)
    name = _english(p.get("name"))
    if name.endswith(".title"):
        return name[: -len(".title")]
    return name


def _run_state(state: dict[str, Any], character: str) -> dict[str, Any]:
    """Headless ``player`` summary -> the bridge ``run_state`` envelope shape
    that ``realgame_ppo._base_obs``/``_base_metrics`` read."""
    player = state.get("player") or {}
    master_deck = [
        {
            "id": _plain_card_id(c.get("id")) or str(c.get("name") or ""),
            "upgrade_level": 1 if c.get("upgraded") else 0,
        }
        for c in (player.get("deck") or [])
    ]
    potions = [
        # Bridge slots carry {slot, empty, id}; headless lists only the
        # FILLED potions ({index, name, ...}).
        {"slot": p.get("index", i), "empty": False, "id": _potion_obs_id(p)}
        for i, p in enumerate(player.get("potions") or [])
    ]
    # `potion_slots` (total belt capacity) lets us emit the empty slots
    # too, exactly like the bridge — realgame_ppo._build_shop's belt-full
    # filter (the shop buy_potion TooFull loop fix) keys off `empty`.
    # Fill the slots NOT already claimed by a filled potion (sparse `index`
    # values would otherwise let `range(len(potions), slots)` re-emit an
    # occupied slot, e.g. one potion at index=2 → duplicate slot 2).
    slots = player.get("potion_slots")
    if isinstance(slots, int):
        occupied = {p["slot"] for p in potions}
        for s in range(slots):
            if s not in occupied:
                potions.append({"slot": s, "empty": True, "id": None})
    return {
        "character": character,
        "hp": player.get("hp", 0),
        "max_hp": player.get("max_hp", 0),
        "gold": player.get("gold", 0),
        "relics": [{"id": r.get("id")} for r in (player.get("relics") or [])],
        "potions": potions,
        "master_deck": master_deck,
        "ascension": 0,
    }


def _envelope(state: dict[str, Any], msg_type: str, character: str) -> dict[str, Any]:
    """Common bridge envelope: ``floor`` is the RUN-WIDE TotalFloor and ``act``
    is 1-based, matching live SpireMentor (PROTOCOL.md, SimStateExtractor.cs) --
    these feed the net's ``total_floor``/``act_index`` state features, which
    ran as 0/0 on every engine-world harness decision before this resolution
    (2026-07-16 featurizer audit: the single largest surviving obs drift).

    Producer dialects handled:
      * real DLL: ``context`` dict -- ``act`` 1-based, ``floor`` = ActFloor
        (act-LOCAL), ``total_floor`` run-wide (RunContext emits it since the
        same-day companion fix; older runners fall back to the act-local
        floor, still far better than 0).
      * RustGame csharp emulation: no context dict; top-level ``act``
        (0-based ``rm.act_index``) and ``floor`` (``rm.total_floor()``,
        already run-wide).
    """
    ctx = state.get("context")
    if not isinstance(ctx, dict):
        ctx = {}
    if ctx:
        act = int(ctx.get("act") or 1)
        floor = int(ctx.get("total_floor", ctx.get("floor", 0)) or 0)
    else:
        act = int(state.get("act") or 0) + 1
        floor = int(state.get("floor") or 0)
    return {
        "type": msg_type,
        "floor": floor,
        "act": act,
        "run_state": _run_state(state, character),
    }


def _bridge_map(full_map: dict[str, Any] | None) -> dict[str, Any] | None:
    """``get_map`` payload -> the bridge ``map`` object realgame_ppo reads."""
    if not full_map:
        return None
    rows = full_map.get("rows") or []
    nodes = [n for row in rows for n in row]
    current = next((n for n in nodes if n.get("current")), None)
    cur_coord = full_map.get("current_coord")
    if current is None and isinstance(cur_coord, dict):
        current = cur_coord
    return {
        "nodes": [
            {
                "col": n.get("col"),
                "row": n.get("row"),
                "type": n.get("type"),
                "children": [
                    {"col": c.get("col"), "row": c.get("row")} for c in (n.get("children") or [])
                ],
            }
            for n in nodes
        ],
        "current": ({"col": current.get("col"), "row": current.get("row")} if current else None),
        "boss": full_map.get("boss"),
        "visited": [{"col": n.get("col"), "row": n.get("row")} for n in nodes if n.get("visited")],
    }


def to_bridge_map_select(
    state: dict[str, Any], full_map: dict[str, Any] | None, character: str
) -> dict[str, Any]:
    out = _envelope(state, "map_select", character)
    choices = state.get("choices") or []
    out["nodes"] = [
        {
            "index": i,
            "type": c.get("type"),
            "row": c.get("row"),
            "col": c.get("col"),
        }
        for i, c in enumerate(choices)
    ]
    out["map"] = _bridge_map(full_map)
    return out


def to_bridge_card_reward(state: dict[str, Any], character: str) -> dict[str, Any]:
    out = _envelope(state, "card_reward", character)
    out["cards"] = [
        {
            "id": _plain_card_id(c.get("id")),
            "name": _english(c.get("name")),
            "rarity": c.get("rarity"),
            "upgrade_level": 1 if c.get("upgraded") else 0,
        }
        for c in (state.get("cards") or [])
    ]
    out["can_skip"] = bool(state.get("can_skip"))
    return out


def to_bridge_rest_site(state: dict[str, Any], character: str) -> dict[str, Any]:
    out = _envelope(state, "rest_site", character)
    out["options"] = [
        {
            "index": o.get("index", i),
            "id": o.get("option_id"),
            "action": "rest_option",
            "label": _english(o.get("name")) or o.get("option_id"),
            "enabled": bool(o.get("is_enabled", True)),
        }
        for i, o in enumerate(state.get("options") or [])
    ]
    return out


#: Act-start Ancient events whose 1-of-3 relic offer IS the act's boss-relic
#: choice (Hive: Orobas/Pael/Tezcatara, decomp Acts/Hive.cs:26-28; Glory:
#: Nonupeipe/Tanx/Vakuu, Acts/Glory.cs:25-27). The Rust engine hoists this
#: offer to ``Phase::BossRelic`` and the PPO net was TRAINED on the
#: ``boss_relic`` surface (scores relics by identity); live SpireMentor emits a
#: real ``boss_relic`` message too. The headless DLL instead rides it on the
#: Ancient's generic ``event_choice`` (RunSimulator.cs:3483-3495), and the
#: EVENT surface never learned identity (event vocab 100% OOV at training) —
#: the net's pick is effectively positional (2026-07-15 ablation: delegating
#: JUST this decision, ~1-2 per run, collapsed the native 10/20 baseline).
#: NEOW is deliberately absent: natively it IS an event, labeled with the
#: blessing enum ids (`_event_option_label`).
_BOSS_RELIC_ANCIENTS = frozenset({"OROBAS", "PAEL", "TEZCATARA", "NONUPEIPE", "TANX", "VAKUU"})


def _boss_relic_reroute(state: dict[str, Any], character: str) -> dict[str, Any] | None:
    """A boss-relic-offer ``event_choice`` -> bridge ``boss_relic`` state.

    Detection is conservative — anything that doesn't look exactly like the
    1-of-N relic offer stays on the event surface: every unlocked option must
    carry a ``text_key`` that is either ``<ACT_START_ANCIENT>.…options.<RELIC>``
    (real-DLL dialect) or a bare relic id with a blank ``event_name`` (the
    RustGame-csharp emulation's ``_fill_boss_relic`` shape), and every
    ``<RELIC>`` tail must resolve in the training relic vocab (full ~290-entry
    ``RelicId`` registry). Dialogue pages ("Continue"), locked options, and
    ordinary events all fail one of these gates and keep today's routing.
    """
    options = state.get("options") or []
    unlocked = [o for o in options if not o.get("is_locked", False)]
    if len(unlocked) < 2:
        return None
    reg = vocab.relics()
    entries: list[tuple[dict[str, Any], str]] = []
    for o in unlocked:
        text_key = str(o.get("text_key") or "")
        prefix, dot, _rest = text_key.partition(".")
        if dot and prefix not in _BOSS_RELIC_ANCIENTS:
            return None
        if not dot and _english(state.get("event_name")):
            return None
        tail = text_key.rsplit(".", 1)[-1]
        if not tail or reg.index(tail) == reg.oov:
            return None
        entries.append((o, tail))
    out = _envelope(state, "boss_relic", character)
    out["options"] = [
        {
            "index": int(o.get("index", i)),
            "relic": {"id": tail, "name": _english(o.get("title"))},
            "enabled": True,
        }
        for i, (o, tail) in enumerate(entries)
    ]
    return out


def to_bridge_event(state: dict[str, Any], character: str) -> dict[str, Any]:
    rerouted = _boss_relic_reroute(state, character)
    if rerouted is not None:
        return rerouted
    out = _envelope(state, "event", character)
    event_id = str(state.get("event_name") or "")
    out["options"] = [
        {
            "index": o.get("index", i),
            "label": _english(o.get("title")),
            "description": _english(o.get("description")),
            "text_key": o.get("text_key"),
            "event_id": event_id,
            # Loc-template values ("Lose {DecipherMaxHpLoss} Max HP" -> 24).
            # Without them `realgame_ppo._resolve_option_vars` no-ops and the
            # encoder's number scrape reads a 24-max-HP cost as free.
            "vars": o.get("vars"),
            "enabled": not o.get("is_locked", False),
            # The world's own lethality verdict (EventOption.WillKillPlayer).
            # Absent from bridges that predate it -> None, which the no-suicide
            # guard reads as "unknown", not as "safe".
            "will_kill_player": o.get("will_kill_player"),
        }
        for i, o in enumerate(state.get("options") or [])
    ]
    return out


def to_bridge_card_select(state: dict[str, Any], character: str) -> dict[str, Any]:
    out = _envelope(state, "card_select", character)
    out["cards"] = [
        {
            "index": c.get("index", i),
            "id": _plain_card_id(c.get("id")),
            "name": _english(c.get("name")),
            "type": c.get("type"),
            "rarity": c.get("rarity"),
            "is_curse": c.get("type") == "Curse",
            "is_status": c.get("type") == "Status",
            "upgrade_level": 1 if c.get("upgraded") else 0,
        }
        for i, c in enumerate(state.get("cards") or [])
    ]
    out["min_count"] = state.get("min_select")
    out["max_count"] = state.get("max_select")
    # Selection semantics from the headless CardSelectCmd context capture
    # (PatchCardSelectContext). No longer drives a heuristic — a mid-combat
    # selection is answered off the persistent combat mirror — but it is the
    # only description of what an OUT-of-combat selection means, and it is how
    # a divergence report names the prompt it fired on.
    out["prompt_key"] = state.get("prompt_key")
    # The card that OPENED the selection. `prompt_key` alone cannot classify a
    # mid-combat selection: all 23 v0.107.1 combat cards that open one pass
    # `base.SelectionScreenPrompt`, a per-card loc key, not a `TO_*` key.
    out["source_model"] = state.get("source_model")
    # `CardSelectorPrefs.Cancelable` verbatim when the world supplies it. The
    # old `min_select <= 0` guess was wrong in BOTH directions: Cook/Smith and
    # RewardSynchronizer set Cancelable with a non-zero MinSelect (a skippable
    # smith read as forced), and FromChooseACardScreen passes min=0 whatever
    # its `canSkip` is (a mandatory obtain read as skippable, letting the agent
    # decline a card the game requires). Kept only as the fallback for an
    # unpatched entry point.
    cancelable = state.get("cancelable")
    if cancelable is None:
        mn = state.get("min_select")
        cancelable = mn is not None and int(mn) <= 0
    out["cancelable"] = bool(cancelable)
    return out


def to_bridge_shop(state: dict[str, Any], character: str) -> tuple[dict[str, Any], list[dict]]:
    """Shop -> the bridge options list + a same-order table mapping each
    option index to its headless ``(action, args)`` command."""
    out = _envelope(state, "shop", character)
    gold = (state.get("player") or {}).get("gold", 0)
    options: list[dict[str, Any]] = [
        {
            "index": 0,
            "id": "leave_shop",
            "action": "leave_shop",
            "label": "Leave shop",
            "enabled": True,
        }
    ]
    commands: list[dict[str, Any]] = [{"action": "leave_room", "args": {}}]

    def add(kind: str, label: str, price: int, item: dict[str, Any], cmd: dict[str, Any]) -> None:
        options.append(
            {
                "index": len(options),
                "id": kind,
                "action": kind,
                "label": label,
                "price": price,
                "enabled": price <= gold,
                "item": item,
            }
        )
        commands.append(cmd)

    for c in state.get("cards") or []:
        if not c.get("is_stocked", True):
            continue
        cid = _plain_card_id(c.get("id")) or _english(c.get("name"))
        add(
            "buy_card",
            _english(c.get("name")),
            int(c.get("cost", 0)),
            {"id": cid, "name": _english(c.get("name"))},
            {"action": "buy_card", "args": {"card_index": c.get("index")}},
        )
    for r in state.get("relics") or []:
        if not r.get("is_stocked", True):
            continue
        add(
            "buy_relic",
            _english(r.get("name")),
            int(r.get("cost", 0)),
            {"id": r.get("id"), "name": _english(r.get("name"))},
            {"action": "buy_relic", "args": {"relic_index": r.get("index")}},
        )
    for p in state.get("potions") or []:
        if not p.get("is_stocked", True):
            continue
        add(
            "buy_potion",
            _english(p.get("name")),
            int(p.get("cost", 0)),
            {"id": p.get("id"), "name": _english(p.get("name"))},
            {"action": "buy_potion", "args": {"potion_index": p.get("index")}},
        )
    removal = state.get("card_removal_cost")
    if isinstance(removal, int) and removal >= 0:
        options.append(
            {
                "index": len(options),
                "id": "remove_card",
                "action": "remove_card",
                "label": "remove_card",
                "price": removal,
                "enabled": removal <= gold,
            }
        )
        commands.append({"action": "remove_card", "args": {}})
    out["options"] = options
    out["purge_cost"] = removal
    return out, commands


# ---------------------------------------------------------------------------
# Combat: native headless picker (mirrors realgame_mcts.pick_combat_action_mcts).
# ---------------------------------------------------------------------------


def pick_combat_action_headless(
    state: dict[str, Any],
    cfg: RecommendConfig,
    rng: random.Random,
    *,
    sims: int | None = None,
    banned: set[str] | None = None,
    mirror: CombatMirror | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """One combat decision via MCTS on the raw headless ``combat_play``.

    Returns ``(headless command, log_extra, engine action)``. ``log_extra`` is
    non-None only when MCTS failed and the random-legal fallback answered
    (counted as a fallback by the session loop, same contract as live).
    ``engine action`` is the same choice in the engine's own dialect, or None
    when no mirror is driving; the caller applies it AFTER the DLL has
    accepted the command, so a rejected play never advances the mirror.

    ``banned`` holds sort-keyed JSON of commands the DLL already REJECTED for
    this exact state (returned an ``error`` state; e.g. the seed-1025 wedge:
    ``can_play`` was true on the wire yet every play bounced). The ranking
    walk skips them so the search's next-best action gets a try instead of
    the deterministic policy re-sending the rejected play forever.

    ``mirror`` is the fight's persistent :class:`CombatMirror`. When present
    the state is reconciled against it and the search runs on a CLONE of it,
    so the search sees the RNG cursors and draw-pile pins that a per-decision
    reconstruction throws away — and the mirror ends the decision parked on
    whatever choice the played card opened, which is what makes the following
    ``card_select`` answerable at all.
    """
    hand = state.get("hand") or []
    run_cfg = cfg if sims is None else RecommendConfig(**{**cfg.__dict__, "sims": sims})
    # The mirror's search targets by encounter SLOT, corpses kept in place;
    # the DLL numbers living enemies only. None: the search ran on the wire
    # state, which lists the living first, so the two agree.
    slots: list[dict[str, Any]] | None = None

    try:
        if mirror is not None:
            wire = adapter.translate(
                state,
                character=run_cfg.character,
                combat_id=run_cfg.combat_id,
                seed=run_cfg.seed,
                ascension=run_cfg.ascension,
            )
            mirror.reconcile(wire)
            tags = tags_from_result(mirror.search(run_cfg), wire["deck"]["hand"])
            slots = _mirror_encounter(mirror)
        else:
            tags = recommend_all(state, run_cfg)
    except MirrorDivergence:
        # STRICT means STRICT. The handler below exists for a search that
        # FAILED — a crashed rollout, a state the recommender could not build —
        # and its fallback is to play a random legal card. A
        # `MirrorDivergence` is the opposite kind of event: the mirror has
        # PROVEN it disagrees with reality, and the whole point of STRICT is
        # that the run stops there with a forensic dump rather than degrading
        # quietly. It is a `RuntimeError`, so the blanket `except Exception`
        # used to swallow it and STRICT never fired on `combat_play` at all —
        # contradicting this module's docstring and every "clean measurement"
        # claim `RunOutcome.mirror_stats` makes. Measured on the first real
        # STRICT sweep: 60 runs reported 30 "clean" completions while
        # `mirror_stats.desyncs` said only 4 of them had never diverged.
        #
        # Losing the crash was the smaller half. `reconcile` assigns
        # `self._state = fresh` BEFORE reporting a desync, and the fallback
        # below returns `engine_action=None`, so the session loop skips
        # `mirror.advance(...)` — the world then commits an action the mirror
        # never saw and every later reconcile reports cascading FALSE
        # divergences on top of the real one. On the transplant-refusal path
        # the mirror keeps its STALE carried state instead. Either way the
        # search runs against a board known to be wrong.
        raise
    except Exception as e:
        logger.warning("headless combat MCTS failed: %s", e)
        playable = [c for c in hand if c.get("can_play")]
        if playable:
            c = rng.choice(playable)
            args: dict[str, Any] = {"card_index": c["index"]}
            if c.get("target_type") == "AnyEnemy" and (state.get("enemies") or []):
                args["target_index"] = 0
            return {"action": "play_card", "args": args}, {"mcts_error": str(e)}, None
        return {"action": "end_turn", "args": {}}, {"mcts_error": str(e)}, None

    def _allowed(cmd: dict[str, Any]) -> bool:
        return not banned or json.dumps(cmd, sort_keys=True) not in banned

    def _dll_target(target: int) -> int:
        return target if slots is None else _shadow_target_to_dll(slots, int(target))

    by_index = {int(c.get("index", i)): c for i, c in enumerate(hand)}
    for tag in tags:  # best-first; skip plays the game itself vetoes
        si = tag.stable_idx
        if si == END_TURN_STABLE_IDX:
            return {"action": "end_turn", "args": {}}, None, {"action": "end_turn"}
        if si <= POTION_STABLE_IDX_BASE:
            slot = POTION_STABLE_IDX_BASE - si
            args = {"potion_index": slot}
            if tag.target_index is not None:
                args["target_index"] = _dll_target(tag.target_index)
            cmd = {"action": "use_potion", "args": args}
            if not _allowed(cmd):
                continue
            return (
                cmd,
                None,
                {"action": "potion", "slot": slot, "target_index": tag.target_index},
            )
        if si < 0:
            continue  # channel/evoke etc. — no headless verb mapped yet
        card = by_index.get(si)
        if card is None or not card.get("can_play", False):
            continue
        args = {"card_index": si}
        if tag.target_index is not None:
            args["target_index"] = _dll_target(tag.target_index)
        cmd = {"action": "play_card", "args": args}
        if not _allowed(cmd):
            continue
        # The engine indexes the HAND; the DLL indexes its own stable card
        # handles. They coincide here because the adapter stamps `stable_idx`
        # from the C# `card.index` and the search ran on that same hand.
        engine_index = next(
            (i for i, c in enumerate(hand) if int(c.get("index", i)) == si),
            None,
        )
        if engine_index is None:
            continue
        return (
            cmd,
            None,
            {
                "action": "play",
                "card_index": engine_index,
                "target_index": tag.target_index,
            },
        )
    return {"action": "end_turn", "args": {}}, None, {"action": "end_turn"}


def pick_card_select_mirrored(
    state: dict[str, Any],
    cfg: RecommendConfig,
    mirror: CombatMirror,
    *,
    sims: int | None = None,
) -> tuple[dict[str, Any], list[int] | None]:
    """Answer a MID-COMBAT ``card_select`` off the persistent mirror.

    The mirror already executed the card that opened this screen, so it is
    parked on the engine's own version of the choice. This reconciles the
    board the DLL attached, re-opens the choice over the WORLD's option list
    (demoting the engine's derivation to an assertion — see
    :meth:`CombatMirror.inject_choice`), searches it, and returns the picks.

    This replaces the three-tier chain outright. There is no replay to fail,
    no prompt-key heuristic to guess with, and no random fallback: a board the
    engine cannot account for raises instead.

    Returns:
        ``(headless command, picked option indices)``. The caller applies the
        picks to the mirror only after the DLL accepts the command.
    """
    board = state["combat"]
    run_cfg = cfg if sims is None else RecommendConfig(**{**cfg.__dict__, "sims": sims})
    wire = adapter.translate(
        board,
        character=run_cfg.character,
        combat_id=run_cfg.combat_id,
        seed=run_cfg.seed,
        ascension=run_cfg.ascension,
    )
    mirror.reconcile(wire)

    parked = mirror.parked_choice()
    options = [
        {
            "id": adapter.registry_card_id(str(c.get("id")), f"$.cards[{i}]"),
            "upgraded": bool(c.get("upgraded")),
        }
        for i, c in enumerate(state.get("cards") or [])
    ]
    # Clamp the arity to what is actually on screen. "Choose any number"
    # arrives as a sentinel max (the C# selector reports int.MaxValue), which
    # is both meaningless past the option count and unrepresentable in the
    # engine's u8 arity fields.
    n = len(options)
    min_select = max(0, min(int(state.get("min_select") or 0), n))
    max_select = max(min_select, min(int(state.get("max_select") or 1), n))
    # A `FromChooseACardScreen` reports min 0 whether or not it may be
    # declined; the truth is the separate `canSkip` flag. Normalize BEFORE the
    # recognition, which keys the Gambling-Chip mulligan on `min_select == 0`.
    can_skip, min_select = skip_arity_from_world(
        cancelable=state.get("cancelable"),
        source=state.get("source"),
        min_select=min_select,
        max_select=max_select,
    )
    # Kind derivation + unparked-screen recognition (generative screens,
    # KnowledgeDemon's curse pick, Gambling Chip's turn-1 mulligan) is SHARED
    # with the live SpireMentor driver (`realgame_mcts.pick_card_select_mirrored`)
    # — see `combat_mirror.recognize_card_select` for the full rationale.
    # `turn_start_blocked_on` names the relic OR power whose turn-start hook
    # the world is blocked inside, when the screen is recognized. Consumed
    # after `inject_choice` to park the rest of turn setup behind the pick —
    # see `CombatMirror.arm_turn_start_resume_for`.
    kind, turn_start_blocked_on = recognize_card_select(
        parked=parked,
        options=options,
        board=board,
        wire=wire,
        min_select=min_select,
        source_model=state.get("source_model"),
        prompt_key=state.get("prompt_key"),
    )
    mirror.inject_choice(
        kind,
        options,
        min_select,
        max_select,
        can_skip=can_skip,
        origin={
            "prompt_key": state.get("prompt_key"),
            "source": state.get("source"),
            "source_model": state.get("source_model"),
        },
    )
    if turn_start_blocked_on is not None:
        # C# `Hook.AfterPlayerTurnStart` AWAITS the screen it opened — Gambling
        # Chip's turn-1 mulligan, or (GH #355, any turn) Tyranny's exhaust /
        # Tools of the Trade's discard — so the
        # wire this mirror was just seeded from is a mid-turn-setup snapshot:
        # the hand is drawn and energy is reset, but MercuryHourglass's 3-to-all
        # (same hook, later in the equip order), BloodVial's
        # `AfterPlayerTurnStartLate` heal and the whole `AfterSideTurnStart` walk
        # (Akabeko's Vigor 8, Lantern's +1 energy) have NOT run. Park that batch
        # behind the pick so resolving the injected choice runs it in C# order —
        # without this the mirror simply never ran it and stayed short for the
        # rest of the combat (18 v34wr records, seeds 1004 / 1047). Armed BEFORE
        # the search so the rollouts see the same continuation.
        mirror.arm_turn_start_resume_for(turn_start_blocked_on)

    if not mirror.alive:
        # OBSERVE only: the injection itself diverged and the mirror was
        # dropped, so there is no state left to search. Take the minimum
        # legal selection and let the run continue TAGGED — STRICT never
        # reaches this line, and no clean measurement may come from a run
        # that did.
        #
        # Do NOT re-seed here: `wire` is the PRE-choice board, and the world
        # is about to resolve the very screen the mirror could not answer. A
        # mirror seeded from it reads the whole resolution as a divergence
        # one decision later (50 of the 230 player-side events on the v28wr
        # census were exactly this — Akabeko's turn-start VIGOR "appearing"
        # against a mirror seeded one instant too early). Stay dropped;
        # `reconcile` seeds unconditionally from the next decision's
        # POST-choice wire, which diverges zero times.
        #
        # Picks are `None`, not `[]`: the caller cannot tell an empty list
        # from a legitimate skip, and would answer it with
        # `advance_through_choice([])` against a mirror that has no choice
        # open — a second, spurious divergence stacked on the real one.
        picks = list(range(min_select))
        if not picks:
            return {"action": "skip_select", "args": {}}, None
        return (
            {"action": "select_cards", "args": {"indices": ",".join(str(i) for i in picks)}},
            None,
        )

    # One search per pick. The engine decomposes a multi-pick sequentially, so
    # each pick is now chosen with the previous ones already in the root state
    # (a single-pick screen is still exactly one search). Policy quality only:
    # the picks are applied to the mirror atomically either way, so state
    # fidelity does not depend on it. The shared method still treats the
    # search's FINALIZE/SKIP row as the stop signal and never emits a
    # duplicate index — the DLL's `ResolvePendingByIndices` keeps duplicates,
    # and Gambling Chip's `DiscardAndDraw(list, list.Count)` then draws one
    # more card than it discards (the mirror_d1 "-1 hand" family, seeds
    # 1004/1047).
    picks = mirror.search_choice_picks(
        run_cfg,
        hand=wire["deck"]["hand"],
        n_options=len(options),
        min_select=min_select,
        max_select=max_select,
    )
    if not picks and min_select <= 0:
        return {"action": "skip_select", "args": {}}, []
    if not picks:
        picks = [0]
    return (
        {"action": "select_cards", "args": {"indices": ",".join(str(i) for i in picks)}},
        picks,
    )


# ---------------------------------------------------------------------------
# Seed-locked shadow engine (Phase 1 of the shadow-engine plan).
# ---------------------------------------------------------------------------


def _shadow_act_args(command: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Translate a DLL-dialect session command to the shadow ``act`` call.

    The shadow ``RustGame(target_convention="csharp")`` speaks the same DLL
    verb dialect the session sends to the real game, so forwarding is a
    passthrough — with the one census carve-out: a confirm-empty
    ``select_cards {indices: ""}`` (answering a "choose up to N" with
    nothing) is modelled by the emulation as ``skip_select``.
    """
    verb = command["action"]
    args = dict(command.get("args") or {})
    if verb == "select_cards" and not str(args.get("indices", "")).strip():
        return "skip_select", {}
    return verb, args


def _mirror_encounter(mirror: CombatMirror) -> list[dict[str, Any]]:
    """The mirror's encounter in its own slot order (corpses kept in place)."""
    return sts2_rs.state_to_dict(mirror.state).get("encounter") or []


def _shadow_target_to_dll(encounter: list[dict[str, Any]], native_target: int) -> int:
    """Map a native-search ``target_index`` to the DLL's alive-enemy frame.

    The native ``mcts_recommend`` targets by SLOT within the round-trip's
    encounter — escaped creatures dropped, but DEAD corpses retained in
    their slots (probe seed 1021: a dead TwigSlime keeps slot 0, so the two
    live slimes are targeted as 1 and 2). The DLL's ``play_card`` resolves
    ``target_index`` against ``Enemies.Where(IsAlive)`` (RunSimulator.cs:1736)
    — an ALIVE-compact list. So slot ``native_target`` becomes the count of
    alive (non-dead, non-escaped) creatures ahead of it. Without this remap a
    play targeting the wrong slot silently lands on a different enemy — a
    mirrored HP swap between two enemies (the tell that surfaced this).
    """
    kept = [e for e in encounter if not e.get("is_escaped")]  # state_from_dict frame
    if not 0 <= native_target < len(kept):
        return native_target
    return sum(1 for e in kept[:native_target] if not e.get("is_dead"))


def pick_combat_action_from_shadow(
    shadow: RustGame,
    dll_state: dict[str, Any],
    cfg: RecommendConfig,
    rng: random.Random,
    *,
    sims: int | None = None,
    banned: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
    """One combat decision searched off the shadow's LOSSLESS native state.

    The shadow engine holds the true combat state (private counters, draw
    order, pending rolls) because it executed the same actions that created
    them. We search that instead of the DLL's lossy wire reconstruction:
    round-trip the native combat state through ``state_to_dict ->
    state_from_dict`` (the rung-3 channel — this deliberately DISCARDS the
    true RNG cursors so the search stays honest, never clairvoyant) and run
    the same ``mcts_recommend`` the wire path uses, with the SAME config, so
    the only variable vs :func:`pick_combat_action_headless` is input
    fidelity.

    Returns ``(command, None)`` on a shadow-searched decision (``log_extra``
    is None — this is not a random fallback), or ``None`` to signal the
    caller to fall back to the wire path (no live combat state, a pending
    in-combat choice the round-trip can't carry, or an MCTS error).

    The recommended action's ``card_index`` is a HAND POSITION in the
    round-tripped state; while the shadow is in sync the hand order is
    identical to the DLL's (verified by the divergence check that gates this
    call), so position ``k`` maps to ``dll_state["hand"][k]`` and its DLL
    ``index`` / ``can_play`` flag. ``target_index`` is passed straight
    through, exactly as the wire path does.
    """
    cs = shadow.combat_state()
    if cs is None or cs.pending_choice() is not None:
        # Not in combat, or a pending card-select the dict round-trip drops
        # (state_to_dict omits pending_choice); the wire path owns selects.
        return None
    try:
        wire = sts2_rs.state_to_dict(cs)
        search_cs = sts2_rs.state_from_dict(wire)
        result = sts2_rs.mcts_recommend(
            search_cs,
            sims=cfg.sims if sims is None else sims,
            c_puct=cfg.resolved_c_puct(),
            depth=cfg.depth,
            seed=cfg.seed,
            threads=cfg.threads,
            fair=cfg.fair,
            rollout_policy=cfg.rollout_policy,
            hidden_info=cfg.hidden_info,
            draw_widening_c=cfg.resolved_draw_widening_c(),
        )
    except Exception as e:
        logger.warning("shadow combat MCTS failed: %s - falling back to wire", e)
        return None

    dll_hand = dll_state.get("hand") or []
    encounter = wire.get("encounter") or []

    def _allowed(cmd: dict[str, Any]) -> bool:
        return not banned or json.dumps(cmd, sort_keys=True) not in banned

    actions = sorted(result["actions"], key=lambda e: (not e["is_best"], -float(e["p_win"])))
    for entry in actions:  # best-first; skip plays the DLL itself vetoes
        act = entry["action"]
        kind = act.get("action")
        if kind == "end_turn":
            return {"action": "end_turn", "args": {}}, None
        if kind in ("potion", "drink_potion"):
            args: dict[str, Any] = {"potion_index": int(act["slot"])}
            if act.get("target_index") is not None:
                args["target_index"] = _shadow_target_to_dll(encounter, int(act["target_index"]))
            cmd = {"action": "use_potion", "args": args}
            if not _allowed(cmd):
                continue
            return cmd, None
        if kind != "play":
            continue  # channel/evoke/choose: no headless verb — skip
        k = int(act["card_index"])
        if not 0 <= k < len(dll_hand):
            continue
        card = dll_hand[k]
        if not card.get("can_play", False):
            continue
        args = {"card_index": int(card.get("index", k))}
        if act.get("target_index") is not None:
            args["target_index"] = _shadow_target_to_dll(encounter, int(act["target_index"]))
        cmd = {"action": "play_card", "args": args}
        if not _allowed(cmd):
            continue
        return cmd, None
    return {"action": "end_turn", "args": {}}, None


# ---------------------------------------------------------------------------
# Bridge-dialect answer -> headless command.
# ---------------------------------------------------------------------------


def _bridge_answer_to_command(
    msg_type: str,
    answer: dict[str, Any],
    state: dict[str, Any],
    shop_commands: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    kind = answer.get("action")
    if msg_type == "map_select":
        choices = state.get("choices") or []
        if not choices:
            # Boss-transition edge: the DLL emits a map_select with no
            # choices when the only move is the boss node (seed-1015
            # 2026-07-15 sweep died at floor 48 to the resulting
            # IndexError). Proceed instead of crashing the run.
            return {"action": "proceed", "args": {}}
        idx = int(answer.get("index", 0))
        c = choices[idx] if 0 <= idx < len(choices) else choices[0]
        return {"action": "select_map_node", "args": {"col": c["col"], "row": c["row"]}}
    if msg_type == "card_reward":
        if kind == "skip":
            return {"action": "skip_card_reward", "args": {}}
        return {"action": "select_card_reward", "args": {"card_index": int(answer.get("index", 0))}}
    if msg_type == "card_select":
        if kind == "cancel" or kind == "skip":
            return {"action": "skip_select", "args": {}}
        indices = answer.get("indices")
        if indices is None:
            indices = [answer.get("index", 0)]
        return {
            "action": "select_cards",
            "args": {"indices": ",".join(str(int(i)) for i in indices)},
        }
    if msg_type == "shop":
        idx = int(answer.get("index", 0))
        assert shop_commands is not None
        if 0 <= idx < len(shop_commands):
            return shop_commands[idx]
        return {"action": "leave_room", "args": {}}
    if msg_type in ("rest_site", "event", "boss_relic"):
        # boss_relic: `_boss_relic_reroute`d Ancient relic offers answer with
        # the ORIGINAL event option index, so choose_option maps back 1:1.
        return {"action": "choose_option", "args": {"option_index": int(answer.get("index", 0))}}
    if msg_type == "bundle_select":
        # Scroll Boxes: pick 1 of N card packs. No trained surface — the
        # answer comes from realgame_random's best-effort ``choose``/0, and
        # any in-range pack index is a legal answer. Mapping it to
        # ``proceed`` (the old fallthrough) left the bundle pending forever:
        # the state never changed and the stall breaker aborted the run at
        # floor 1 (seed 1016, sweeps 20260717T012146Z / 20260717T041112Z).
        bundles = state.get("bundles") or []
        idx = int(answer.get("index", 0))
        if not 0 <= idx < len(bundles):
            idx = 0
        return {"action": "select_bundle", "args": {"bundle_index": idx}}
    return {"action": "proceed", "args": {}}


# ---------------------------------------------------------------------------
# Session loop.
# ---------------------------------------------------------------------------


def play_one_headless_session(
    policy_rng: random.Random,
    *,
    run_index: int,
    character: str,
    seed: int,
    mcts_cfg: RecommendConfig,
    ppo_policies: PpoPolicyCache,
    sims_regular: int = DEFAULT_SIMS_REGULAR,
    sims_boss: int = DEFAULT_SIMS_BOSS,
    max_steps: int = 3000,
    max_run_seconds: float = 1800.0,
    jsonl: JsonlLogger | None = None,
    played_save: bool = False,
    shadow: bool = False,
    trace_path: Path | None = None,
    forward_stderr: bool = False,
    ascension: int = 0,
    game: Any = None,
) -> RunOutcome:
    """Play one run on the C# host with the live decision stack.

    Args:
        ascension: Ascension level the run starts at.
        game: A ``sts2_client.Game``-shaped world to play instead of a fresh
            host process. ``live_game_diff.LockstepGame`` passes itself here
            to play the host and the real game in lockstep (GH #1287). It
            must expose ``start``, ``send``, ``get_map``, ``proc`` and
            ``world_errors`` like ``Game``.
    """
    start = time.monotonic()
    step = 0
    fallback_count = 0
    result = "unknown"
    floor: int | None = None
    act: int | None = None
    error: str | None = None
    session: dict[str, Any] = {}
    current_room_type: str | None = None
    full_map: dict[str, Any] | None = None
    won = False
    act_floors: dict[int, int] = {}
    last_cmd_key: tuple | None = None
    repeat_count = 0
    last_state_key: int | None = None
    stall_count = 0
    recent_keys: deque[int] = deque(maxlen=_STALL_WINDOW)
    window_counts: Counter[int] = Counter()
    banned_for_state: dict[int, set[str]] = {}
    prev_state_key: int | None = None
    prev_command: dict[str, Any] | None = None
    prev_decision: Any = None

    # Seed-locked shadow engine: a lockstep RunManager (csharp-dialect
    # emulation) started from the same string seed and fed every command the
    # DLL gets. While it stays in sync with the DLL wire, combat is searched
    # off its lossless native state; the first divergence retires it and the
    # run falls back to the wire path for the rest — strictly additive fidelity.
    shadow_backend: RustGame | None = None
    shadow_alive = shadow
    shadow_divergence: dict[str, Any] | None = None
    shadow_alive_floors = 0
    shadow_combat_searches = 0
    shadow_prev_decision: Any = None  # for the shop_fresh carve-out
    tracer = WireRunTracer(trace_path, seed) if trace_path is not None else None

    # Bound before the try so the finally's `tracer.finish(state, ...)` never
    # trips NameError (and silently leaks the trace handle) when game.start()
    # raises before `state` is first assigned.
    state: dict[str, Any] = {}
    # One persistent engine state per fight (plan Phase 2). It replaces the
    # per-decision reconstruction outright: combat searches run on a clone of
    # it, and a mid-combat card_select is answered off the choice the played
    # card already opened on it. `STS2_MIRROR_OBSERVE=1` downgrades a
    # divergence from a crash to a tagged, logged drop.
    mirror = CombatMirror(
        mcts_cfg,
        policy=policy_from_env(),
        label=f"headless-{character}-{seed}",
    )
    if game is None:
        game = Game(forward_stderr=forward_stderr)
    world_errors: dict[str, Any] = {"count": 0, "sample": []}
    try:
        state = game.start(character, seed=str(seed), ascension=ascension, played_save=played_save)
        if shadow:
            shadow_backend = RustGame(target_convention="csharp")
            shadow_backend.start(character=character, seed=str(seed), ascension=0)
        reset_drop_counts()  # scope the adapter soft-drop tally to this run
        while step < max_steps:
            # Wall-clock backstop for stall variants the byte-identical check
            # misses (e.g. a counter mutating in the echoed state). Legitimate
            # runs finish in <=~6 min; the seed-1007/1009 livelocks ran ~90.
            if time.monotonic() - start > max_run_seconds:
                result = "time_cap_reached"
                logger.warning(
                    "run %d step %d: exceeded %.0fs wall clock - aborting run",
                    run_index,
                    step,
                    max_run_seconds,
                )
                break
            state_key = hash(json.dumps(state, sort_keys=True, default=str))
            if state_key == last_state_key:
                stall_count += 1
                if stall_count >= _STALL_REPEAT_LIMIT:
                    result = "stall_detected"
                    logger.warning(
                        "run %d step %d: state unchanged %d steps in a row - aborting stalled run",
                        run_index,
                        step,
                        stall_count,
                    )
                    break
            else:
                stall_count = 0
                last_state_key = state_key
            # Cyclic-stall breaker: the wedged DLL can ALTERNATE two states
            # (combat state <-> rejection "error" state), so the consecutive
            # check above never accumulates. Count occurrences of each state
            # inside a sliding window instead (seed 1025, 2026-07-18 sweep).
            if len(recent_keys) == recent_keys.maxlen:
                window_counts[recent_keys[0]] -= 1
            recent_keys.append(state_key)
            window_counts[state_key] += 1
            if window_counts[state_key] >= _STALL_REPEAT_LIMIT:
                result = "stall_detected"
                logger.warning(
                    "run %d step %d: state recurred %d times in the last %d steps - "
                    "aborting cyclically stalled run",
                    run_index,
                    step,
                    window_counts[state_key],
                    len(recent_keys),
                )
                break
            decision = state.get("decision") or state.get("type")
            # The host's wire does not name its build; the bridge's envelope
            # does. Without it the combat search rebuilds the fight as the
            # default build and rejects a v0.111.0 run seed (GH #1287).
            if state.get("type") == "decision":
                state.setdefault("game_version", sts2_client.GAME_VERSION)
                # A mid-combat card_select carries its fight nested; the
                # mirror rebuilds from that dict.
                if isinstance(state.get("combat"), dict):
                    state["combat"].setdefault("game_version", sts2_client.GAME_VERSION)
            if state.get("type") == "error":
                # Errors carry no decision; nudge forward like the random
                # driver. First attribute the rejection: ban the command that
                # produced this error for the state it was picked in, so the
                # next visit re-picks the search's next-best action instead of
                # re-sending the rejected play forever (seed-1025 wedge: the
                # DLL said ``can_play`` on the wire, then bounced every play).
                if (
                    prev_decision == "combat_play"
                    and prev_command is not None
                    and prev_state_key is not None
                ):
                    banned_for_state.setdefault(prev_state_key, set()).add(
                        json.dumps(prev_command, sort_keys=True)
                    )
                    logger.warning(
                        "run %d step %d: DLL rejected %s (%s) - banning for this state",
                        run_index,
                        step,
                        prev_command.get("action"),
                        state.get("message") or state.get("error") or "no detail",
                    )
                if shadow_alive:
                    # The DLL wedged and we're about to nudge it with a
                    # "proceed" the shadow never saw — the command streams
                    # desync from here, so retire the shadow.
                    shadow_alive = False
                    shadow_divergence = {
                        "step": step,
                        "floor": floor,
                        "decision": prev_decision,
                        "kind": "dll_error",
                        "diffs": [{"field": "dll_state", "wire": "error", "shadow": "alive"}],
                    }
                    logger.info("run %d step %d: shadow retired (DLL error path)", run_index, step)
                state = game.act("proceed")
                continue

            ctx = state.get("context") or {}
            # Headless `floor` is PER-ACT (1..~17, resets each act), unlike the
            # bridge's global floor. Accumulate the per-act maxima so the
            # reported floor is comparable with live sweeps (act3 boss ~ 48).
            if isinstance(ctx.get("floor"), int) and isinstance(ctx.get("act"), int):
                act_floors[ctx["act"]] = max(act_floors.get(ctx["act"], 0), ctx["floor"])
                floor = sum(act_floors.values())
                act = max(act_floors) if act is None else max(act, ctx["act"])
            if tracer is not None:
                tracer.observe(state)

            if decision in _TERMINAL_DECISIONS or state.get("type") in (
                "game_over",
                "run_complete",
            ):
                step += 1
                won = bool(state.get("victory") or state.get("player_won"))
                result = "victory" if won else "defeat"
                if jsonl is not None:
                    jsonl.log_terminal(step, state)
                if tracer is not None:
                    tracer.finish(state, won)
                    tracer = None
                break

            # Shadow divergence check: both worlds are now at the SAME
            # decision point (the shadow was advanced with every prior
            # command). While its serialized state still matches the DLL
            # wire on the overlapping surface, combat is searched off its
            # lossless native state; the first hard diff retires it.
            if shadow_alive and shadow_backend is not None:
                shop_fresh = not (decision == "shop" and shadow_prev_decision == "shop")
                try:
                    sh_state = shadow_backend.get_state()
                    diffs = [
                        d
                        for d in diff_states(state, sh_state, shop_fresh=shop_fresh)
                        if not d.get("soft")
                    ]
                except Exception as e:  # shadow serialization failed — retire it
                    diffs = [{"field": "shadow_error", "wire": None, "shadow": repr(e)}]
                if diffs:
                    shadow_alive = False
                    shadow_divergence = {
                        "step": step,
                        "floor": floor,
                        "act": act,
                        "decision": decision,
                        "diffs": diffs[:20],
                    }
                    logger.info(
                        "run %d step %d: shadow diverged at floor %s (%s: %s)",
                        run_index,
                        step,
                        floor,
                        decision,
                        ", ".join(sorted({d["field"] for d in diffs})),
                    )
                else:
                    shadow_alive_floors = floor or shadow_alive_floors
                if decision != "card_select":
                    shadow_prev_decision = decision

            command: dict[str, Any]
            log_extra: dict[str, Any] | None = None
            shop_commands: list[dict[str, Any]] | None = None

            # The persistent combat mirror follows the fight, so it only
            # survives while the DLL is still showing one. Anything else ends
            # it — including an out-of-combat card_select (a rest-site upgrade
            # carries no board), which the PPO path answers as before.
            in_fight = decision == "combat_play" or (
                decision == "card_select" and state.get("combat") is not None
            )
            if not in_fight:
                mirror.drop("left combat")

            pending_picks: list[int] | None = None
            engine_action: dict[str, Any] | None = None

            if decision == "combat_play":
                boss = current_room_type in ("boss", "Boss")
                sims = sims_boss if boss else sims_regular
                command = log_extra = None  # type: ignore[assignment]
                if shadow_alive and shadow_backend is not None:
                    picked = pick_combat_action_from_shadow(
                        shadow_backend,
                        state,
                        mcts_cfg,
                        policy_rng,
                        sims=sims,
                        banned=banned_for_state.get(state_key),
                    )
                    if picked is not None:
                        command, log_extra = picked
                        shadow_combat_searches += 1
                        # The shadow searched its own lossless state, so the
                        # mirror never saw this decision and cannot know what
                        # the world just did. Drop the carried state rather
                        # than let it answer from a board it did not track.
                        #
                        # This is a DROP, not a retirement for the fight:
                        # `reconcile` re-seeds unconditionally when there is
                        # no state, so the mirror is answering again on the
                        # very next combat decision — just from reality
                        # instead of a stale carry, minus the hidden half
                        # (RNG cursors, pins, intent history). That is the
                        # intended recovery; only the continuity is lost.
                        mirror.drop("shadow answered this decision")
                if command is None:
                    command, log_extra, engine_action = pick_combat_action_headless(
                        state,
                        mcts_cfg,
                        policy_rng,
                        sims=sims,
                        banned=banned_for_state.get(state_key),
                        mirror=mirror,
                    )
                bridge_state: dict[str, Any] = {"type": "combat_action"}
            elif in_fight:
                # Mid-combat selection, answered off the mirror the played
                # card already parked on a choice. No replay, no prompt-key
                # heuristic, no random fallback — see
                # `pick_card_select_mirrored`.
                command, pending_picks = pick_card_select_mirrored(
                    state,
                    mcts_cfg,
                    mirror,
                    sims=sims_boss if current_room_type in ("boss", "Boss") else sims_regular,
                )
                log_extra = None
                bridge_state = {"type": "card_select"}
                session["prev_msg_type"] = "card_select"
            else:
                if decision == "map_select":
                    try:
                        full_map = game.get_map()
                    except Exception:
                        full_map = None
                    bridge_state = to_bridge_map_select(state, full_map, character)
                elif decision == "card_reward":
                    bridge_state = to_bridge_card_reward(state, character)
                elif decision == "rest_site":
                    bridge_state = to_bridge_rest_site(state, character)
                elif decision == "event_choice":
                    bridge_state = to_bridge_event(state, character)
                elif decision == "card_select":
                    bridge_state = to_bridge_card_select(state, character)
                elif decision == "shop":
                    bridge_state, shop_commands = to_bridge_shop(state, character)
                else:
                    # bundle_select / unknown: proceed-style random legality.
                    bridge_state = _envelope(state, str(decision), character)

                # The training obs carry the act map on EVERY decision, but
                # the wire only sends it on map_select -- before the act's
                # first map screen (Neow, act-start Ancient boss relic) the
                # session cache is empty and the net decides MAP-BLIND
                # (2026-07-16 audit: 72 missing map-node tokens flip 20% of
                # Neow picks, the run's most leveraged decision). The DLL
                # exposes get_map, so seed the cache whenever it lacks the
                # current act's graph.
                if bridge_state["type"] != "map_select":
                    cached = session.get("last_map")
                    if not cached or cached[0] != bridge_state.get("act"):
                        try:
                            seeded = _bridge_map(game.get_map())
                        except Exception:
                            seeded = None
                        if seeded and seeded.get("nodes"):
                            session["last_map"] = (bridge_state.get("act"), seeded)
                            session.pop("map_pos", None)  # fresh act: position unknown

                # Only OUT-of-combat card_selects reach here now (rest-site
                # upgrade, event grant, shop removal) — a mid-combat one is
                # answered off the mirror above. This is the trained PPO
                # surface and stays as-is.
                answer, log_extra = pick_noncombat_action_ppo(
                    bridge_state,
                    policy_rng,
                    ppo_policies,
                    run_config_character=character,
                    run_config_seed=seed,
                    session=session,
                )
                msg_type = bridge_state["type"]
                command = _bridge_answer_to_command(msg_type, answer, state, shop_commands)
                if answer.get("annotations"):
                    # Keep the net's probability/value note on the logged
                    # command — proof in the JSONL that PPO answered the
                    # converted surface (vs the silent random fallback).
                    command = {**command, "annotations": answer["annotations"]}
                if msg_type == "map_select":
                    idx = int(answer.get("index", 0))
                    choices = state.get("choices") or []
                    if 0 <= idx < len(choices):
                        current_room_type = str(choices[idx].get("type") or "").lower()
                session["prev_msg_type"] = "event" if msg_type == "event" else msg_type

            if decision == "combat_play":
                session["prev_msg_type"] = "combat_action"

            # Loop breaker: a purchase the game rejects (e.g. buy_potion with a
            # full belt) leaves the state unchanged, and a deterministic policy
            # then re-picks it forever. Mirrors the bridge's MaxPurchaseAttempts.
            key = (
                decision,
                command["action"],
                json.dumps(command.get("args") or {}, sort_keys=True),
            )
            repeat_count = repeat_count + 1 if key == last_cmd_key else 1
            last_cmd_key = key
            if decision == "shop" and repeat_count >= 4 and command["action"] != "leave_room":
                logger.warning(
                    "run %d step %d: shop action %s repeated %d x with no effect - leaving shop",
                    run_index,
                    step,
                    command["action"],
                    repeat_count,
                )
                command = {"action": "leave_room", "args": {}}
                last_cmd_key = None

            if log_extra is not None:
                fallback_count += 1
            step += 1
            if jsonl is not None:
                jsonl.log_step(step, state, command, extra=log_extra)
            prev_state_key = state_key
            prev_command = command
            prev_decision = decision
            state = game.send(
                {
                    "cmd": "action",
                    "action": command["action"],
                    **({"args": command["args"]} if command.get("args") else {}),
                }
            )
            # Advance the mirror only now, and only if the DLL actually took
            # the command: a rejected play (the seed-1025 wedge) never happened
            # in the world, so mirroring it would desync the mirror on purpose.
            if mirror.alive and state.get("type") != "error":
                if engine_action is not None:
                    mirror.advance(engine_action)
                elif pending_picks is not None:
                    mirror.advance_through_choice(pending_picks)
            # Forward the exact command the DLL got to the shadow, keeping the
            # two worlds in lockstep. A shadow-side reject (unported content /
            # dialect gap) retires the shadow; the run continues on the wire.
            if shadow_alive and shadow_backend is not None:
                verb, sh_args = _shadow_act_args(command)
                try:
                    sh_next = shadow_backend.act(verb, **sh_args)
                except Exception as e:
                    sh_next = {"type": "error", "message": repr(e)}
                if isinstance(sh_next, dict) and sh_next.get("type") == "error":
                    shadow_alive = False
                    shadow_divergence = shadow_divergence or {
                        "step": step,
                        "floor": floor,
                        "decision": decision,
                        "kind": "shadow_apply_rejected",
                        "diffs": [
                            {"field": "apply", "wire": command, "shadow": sh_next.get("message")}
                        ],
                    }
                    logger.info(
                        "run %d step %d: shadow retired (apply rejected: %s)",
                        run_index,
                        step,
                        sh_next.get("message"),
                    )
        else:
            result = "max_steps_reached"
    except Exception as e:
        error = str(e)
        result = f"error: {e}"
        logger.error("Run %d failed after %d steps: %s", run_index, step, e)
    finally:
        try:
            game.proc.kill()
            # Killing closes the child's stderr, which EOFs the pipe, which ends the
            # drain loop — so `world_errors()` below joins a thread that has already
            # consumed everything the world wrote. Nothing is lost to the kill:
            # .NET's `Console.Error` is auto-flushing, so a line the child had
            # emitted is in the pipe, not in a userspace buffer that dies with it.
            #
            # A graceful `game.close()` first would look tidier and is deliberately
            # NOT done: it writes `{"cmd":"quit"}` to the child's stdin, and a child
            # wedged badly enough to need this path is exactly one that may never
            # read stdin again — an unbounded write in the one place a sweep cannot
            # afford to block. The `wait` here is bounded for the same reason.
            game.proc.wait(timeout=5)
        except Exception:
            pass
        try:
            world_errors = game.world_errors()
        except Exception:
            pass
        if tracer is not None:  # non-terminal end (stall / error / max-steps)
            try:
                tracer.finish(state, won)
            except Exception:
                pass

    if world_errors.get("count"):
        # WARNING, not INFO: a throw inside the world is swallowed by
        # `TaskHelper.LogTaskExceptions`, so the run keeps going and its result
        # reads like an ordinary loss. This line is the only place it surfaces
        # unless the sweep was run with --forward-stderr.
        logger.warning(
            "run %d seed %d: the C# world threw (%d error line(s)) - this run's result "
            "is SUSPECT (first: %s)",
            run_index,
            seed,
            world_errors["count"],
            (world_errors.get("sample") or ["?"])[0],
        )
    drops = drop_counts()
    if drops.get("total"):
        logger.warning(
            "run %d seed %d: search-state soft-drops %s (MCTS searched a degraded state)",
            run_index,
            seed,
            drops,
        )
    if shadow_backend is not None:
        shadow_backend.close()
    shadow_stats: dict[str, Any] | None = None
    if shadow:
        shadow_stats = {
            "alive_floors": shadow_alive_floors,
            "survived_run": shadow_divergence is None,
            "combat_searches": shadow_combat_searches,
            "divergence": shadow_divergence,
        }
        logger.info(
            "run %d seed %d: shadow alive_floors=%s combat_searches=%d survived=%s%s",
            run_index,
            seed,
            shadow_alive_floors,
            shadow_combat_searches,
            shadow_divergence is None,
            ""
            if shadow_divergence is None
            else f" (diverged floor {shadow_divergence.get('floor')} "
            f"{shadow_divergence.get('decision')})",
        )
    return RunOutcome(
        run_index=run_index,
        policy="ppo",
        seed=seed,
        character=character,
        ascension=ascension,
        result=result,
        won=won,
        floor=floor,
        act=act,
        decisions=step,
        fallback_count=fallback_count,
        duration_s=time.monotonic() - start,
        error=error,
        search_drops=drops or None,
        shadow_stats=shadow_stats,
        mirror_stats={**mirror.stats.as_dict(), "tainted": mirror.tainted},
        # Recorded even at zero. `None` is reserved for "this arm has no world-error
        # instrumentation at all" (the SpireMentor/TCP arm), so that a clean sweep and
        # an uncounted one stay distinguishable in the report — see
        # `realgame_eval._world_err_cell`.
        world_errors=world_errors,
    )


def _sum_search_drops(outcomes: list[RunOutcome]) -> dict[str, int]:
    """Sum per-run adapter soft-drop tallies into one sweep-level dict by kind."""
    total: dict[str, int] = {}
    for o in outcomes:
        for kind, n in (o.search_drops or {}).items():
            total[kind] = total.get(kind, 0) + n
    return total


def _summarize_world_errors(outcomes: list[RunOutcome]) -> dict[str, Any]:
    """Sweep-level tally of C# exceptions, with the seeds that threw.

    Any non-zero ``runs`` here means part of this sweep's non-wins are the
    world failing rather than the policy losing, so the win rate is a floor —
    exactly the trap the SLOTH aborts and the Kaiser Crab arm deaths set.
    """
    thrown = [o for o in outcomes if (o.world_errors or {}).get("count")]
    return {
        "runs": len(thrown),
        "total": sum(int((o.world_errors or {})["count"]) for o in thrown),
        "seeds": [o.seed for o in thrown],
        # One representative line per affected seed — enough to group them by
        # cause without carrying every stack frame into the report.
        "first_lines": {
            str(o.seed): ((o.world_errors or {}).get("sample") or ["?"])[0] for o in thrown
        },
    }


def _summarize_shadow(outcomes: list[RunOutcome]) -> dict[str, Any] | None:
    """Sweep-level shadow telemetry, or None if --shadow wasn't used.

    Reports how often the lockstep shadow survived the whole run, the median
    floors it stayed in sync, and the total combat decisions searched off its
    lossless native state instead of the DLL wire reconstruction.
    """
    runs = [o.shadow_stats for o in outcomes if o.shadow_stats is not None]
    if not runs:
        return None
    floors = sorted(int(s.get("alive_floors") or 0) for s in runs)
    return {
        "runs": len(runs),
        "survived_full_run": sum(1 for s in runs if s.get("survived_run")),
        "median_alive_floors": floors[len(floors) // 2],
        "min_alive_floors": floors[0],
        "max_alive_floors": floors[-1],
        "total_combat_searches": sum(int(s.get("combat_searches") or 0) for s in runs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--sims-regular", type=int, default=DEFAULT_SIMS_REGULAR)
    parser.add_argument("--sims-boss", type=int, default=DEFAULT_SIMS_BOSS)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument(
        "--max-run-seconds",
        type=float,
        default=1800.0,
        help="wall-clock cap per run; a run past this is aborted as "
        "'time_cap_reached' (legit runs finish in <=~6 min, the real-DLL "
        "EndTurn livelocks ran ~90)",
    )
    parser.add_argument(
        "--played-save",
        action="store_true",
        help="use the minimal first-time-player profile (NEOW_EPOCH unrevealed => "
        "no Neow blessing; NumberOfRuns=0 => the fixed weak first-run encounter "
        "tables). Default is the veteran profile (all epochs/encounters unlocked, "
        "Neow fires) so rung-3.5 is a fair twin of the veteran engine world.",
    )
    parser.add_argument(
        "--fresh-save",
        action="store_true",
        help="(deprecated no-op; the veteran profile is now the default)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="PPO model override -- a torch checkpoint (.pt/.pth/.ckpt/.bin/.tar) or an "
        ".onnx export; the extension picks the backend and an unrecognized one is refused. "
        "Failing to load it ABORTS the eval rather than silently scoring the random policy. "
        "Default: the committed ONNX champion per character.",
    )
    parser.add_argument(
        "--clairvoyant",
        action="store_true",
        help="reproduce the OLD leaky combat search (hidden_info=clairvoyant, "
        "c_puct=1.4, draw_widening_c=1.0). Default matches the DEPLOYED honest "
        "MctsCombatPolicy (beliefdraw, c_puct=2.5, draw_widening_c=2.0) that the "
        "engine-world 50%% baseline runs -- rung-3.5 sweeps 1-4 accidentally ran "
        "the clairvoyant config, confounding the combat-vs-content verdict.",
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="run a seed-locked shadow RunManager alongside the DLL and, while "
        "it stays in lockstep, search combat off its LOSSLESS native state "
        "(state_to_dict -> state_from_dict -> mcts_recommend) instead of the "
        "DLL's lossy wire reconstruction. The first divergence retires the "
        "shadow to the wire path for the rest of the run -- strictly additive "
        "fidelity. Reports per-run shadow_alive_floors / combat_searches.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="write a per-seed world-trace JSONL (seed<N>.jsonl; ev:room/combat/"
        "run_end with floor/hp/deck_size) here, matching play_full_run_ppo's "
        "RunTracer schema so the real-DLL and engine-world worlds diff directly "
        "(deck quality + HP at matched floors).",
    )
    parser.add_argument(
        "--forward-stderr",
        action="store_true",
        help="echo the C# process's stderr (prefixed '[sts2]') to ours. Exception "
        "COUNTS are always recorded per run without this flag (report.json "
        "world_errors / the 'world err' report column); use it when you need the "
        "full stack traces, e.g. to root-cause a stall.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("logs/headless_eval"))
    args = parser.parse_args(argv)
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Match the deployed honest combat search (sts2-runai mcts_config_for /
    # BeliefDraw) by default; --clairvoyant restores the pre-fix leaky config
    # for an A/B. See the search-config-confound memory note (2026-07-14).
    mcts_cfg = RecommendConfig(
        character=args.character,
        hidden_info="clairvoyant" if args.clairvoyant else "beliefdraw",
        c_puct=1.4 if args.clairvoyant else 2.5,
        draw_widening_c=None if args.clairvoyant else 2.0,
    )
    ppo_policies = PpoPolicyCache(args.model)
    outcomes: list[RunOutcome] = []
    for i in range(args.runs):
        seed = args.seed_base + i
        jsonl = JsonlLogger(Path("logs/headless"))
        jsonl.log_meta(seed=seed, character=args.character, ascension=0)
        outcome = play_one_headless_session(
            random.Random(seed),
            run_index=i,
            character=args.character,
            seed=seed,
            mcts_cfg=mcts_cfg,
            ppo_policies=ppo_policies,
            sims_regular=args.sims_regular,
            sims_boss=args.sims_boss,
            max_steps=args.max_steps,
            max_run_seconds=args.max_run_seconds,
            jsonl=jsonl,
            played_save=args.played_save,
            shadow=args.shadow,
            trace_path=(args.trace_dir / f"seed{seed}.jsonl") if args.trace_dir else None,
            forward_stderr=args.forward_stderr,
        )
        jsonl.close()
        outcomes.append(outcome)
        logger.info(
            "run %d seed %d: %s floor=%s act=%s decisions=%d fallbacks=%d (%.0fs)",
            i,
            seed,
            outcome.result,
            outcome.floor,
            outcome.act,
            outcome.decisions,
            outcome.fallback_count,
            outcome.duration_s,
        )

    report = build_report(
        outcomes,
        meta={
            "harness": "headless_eval (rung 3.5: real DLL world, Rust search model)",
            "character": args.character,
            "seed_base": args.seed_base,
            "sims_regular": args.sims_regular,
            "sims_boss": args.sims_boss,
            "profile": "played_save (first-timer)" if args.played_save else "veteran (fresh-save)",
            "combat_search": {
                "hidden_info": mcts_cfg.hidden_info,
                "c_puct": mcts_cfg.c_puct,
                "draw_widening_c": mcts_cfg.draw_widening_c,
            },
            # Total adapter soft-drops across the sweep, by kind. Non-zero =>
            # the combat search planned over a degraded state on some runs
            # (a search-fidelity canary; should be 0 for a clean vocab).
            "search_state_drops": _sum_search_drops(outcomes),
            # C# exceptions seen on the DLL's stderr, by run. Non-zero => some
            # of this sweep's losses are the world failing, not the policy.
            "world_errors": _summarize_world_errors(outcomes),
            # Seed-locked shadow-engine telemetry (only when --shadow is set).
            "shadow": _summarize_shadow(outcomes),
        },
    )
    # Suffix the report dir with the PID so concurrent workers (shadow_hp_pilot
    # launches 16, none overriding --out-dir) don't land on the same
    # same-second path and overwrite each other's report.
    ts = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
    json_path, md_path = write_report(report, args.out_dir / ts)
    print(render_markdown(report))
    we = report["meta"]["world_errors"]
    if we["runs"]:
        print(
            f"WARNING: the C# world threw in {we['runs']}/{len(outcomes)} runs "
            f"({we['total']} error lines, seeds {we['seeds']}). Those runs' results are "
            f"NOT clean measurements -- re-run with --forward-stderr for traces."
        )
    print(f"reports: {json_path} / {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
