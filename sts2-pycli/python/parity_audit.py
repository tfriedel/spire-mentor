"""Identity-aware parity audit driver (C# vs Rust), built on parity_fuzz.

The stock parity_fuzz harness compares only pile *counts* and ignores
card *identity* in hand/draw/discard, and its legal-action canonical
form ignores card identity too. Consequence: a draw-order / shuffle RNG
divergence is invisible until it cascades into a damage/hp/block diff,
so the raw STATE buckets conflate real engine bugs with cascade noise.

This driver:
  * extends the canonical state with ordered card-id lists (hand/draw/
    discard/exhaust), so diff() fires at the TRUE first divergence;
  * classifies each first-divergence into a category so the audit can
    separate primary engine bugs from cascade artifacts;
  * attributes the divergence to the card/encounter that caused it.

Categories:
  DRAW_ORDER     first divergence is ordered card-id list, counts equal,
                 hands matched going in, not an upgrade-choice card
                 → shuffle / draw RNG order bug (high value, single root)
  PILE_COUNT     a pile *count* diverges with matching hands going in
                 → a card moved to the wrong pile (exhaust vs discard …)
  PLAYER_HP      player.hp diverges after end_turn, hands matched
                 → monster intent / damage formula bug
  PLAYER_STATE   player.block/energy/powers diverges after a play_card,
                 hands matched → card-effect bug (attributed to the card)
  MONSTER_STATE  monster hp/alive/block/powers diverges, hands matched
                 → card-damage or monster bug (attributed to the card)
  UPGRADE_CHOICE hand_ids diverge right after an upgrade-choice card
                 (ARMAMENTS/…); known-acceptable auto-pick asymmetry
  CASCADE        hands already differed BEFORE this action
                 → downstream of an earlier (undetected) DRAW_ORDER bug
"""

from __future__ import annotations

import argparse
import concurrent.futures as _cf
import dataclasses
import datetime as _dt
import glob
import hashlib
import itertools
import json
import os
import pathlib
import random
import re
import subprocess
import sys
from collections import Counter
from typing import Any

import sts2_client
import sts2_rs  # type: ignore[import-not-found]
from parity_fuzz import (
    _apply_csharp,
    _apply_rust,
    _csharp_init,
    _csharp_legal,
    _drain_csharp_choices,
    _drain_rust_choices,
    _rust_init,
    _rust_legal,
)
from parity_fuzz_scenario import Scenario, coverage_scenarios, generate, generate_v2
from parity_fuzz_state import diff, normalize_csharp, normalize_rust
from sts2_client import Game

UPGRADE_CARDS = {"ARMAMENTS"}  # IC cards that open an upgrade choice

# ---------------------------------------------------------------------------
# Sweep-v2 axis/tier definitions
# ---------------------------------------------------------------------------

V2_AXES = ("upgraded", "arenas", "fuzz", "pairwise", "autoplay", "enchant")
_FUZZ_SEEDS = {"smoke": 5, "quick": 500, "mid": 2000, "full": 5000}
_V2_MAX_TURNS = 6


def assemble_v2_trials(
    tier: str,
    axes: list[str],
    character: str = "Ironclad",
    max_turns: int = _V2_MAX_TURNS,
    paths: int = 1,
) -> list[tuple[int, Any, int]]:
    """Deterministic (seed, scenario, max_turns) list for a v2 sweep.

    Per-trial action seed = position in the assembled list, so a rerun
    with the same tier+axes+character reproduces every trial exactly.

    ``character`` parameterizes the per-class axes (upgraded / arenas /
    fuzz) to the character's card pool. The pairwise axis is Ironclad-only
    (HOOK_GROUPS is an IC-curated hook taxonomy), so it is silently skipped
    for other characters. ``"Ironclad"`` reproduces the original sweep
    byte-for-byte, keeping the committed Ironclad trace cache valid.

    ``max_turns`` (item#3, deep-horizon): how many turns each trial runs.
    Default 6 reproduces the original sweep; raise it (e.g. 30) to reach
    late-combat scaling bugs (Demon Form / Juggernaut / Barricade stacks)
    the 6-turn horizon never reaches. It is part of the trace-cache key, so
    a deeper run records its own traces rather than reusing shallow ones.

    ``paths`` (item#4, multi-path): how many independent action trajectories
    to run per scenario. Each path gets a distinct action seed (the position
    index), so the SAME scenario is explored down ``paths`` different random
    play-lines — surfacing combos/orderings one random path per scenario
    misses. Default 1 reproduces the original 1-path sweep.
    """
    from parity_fuzz_scenario import coverage_upgraded_scenarios

    if tier not in _FUZZ_SEEDS:
        raise SystemExit(f"unknown v2 tier: {tier!r} (valid: {sorted(_FUZZ_SEEDS)})")
    if max_turns < 1:
        raise SystemExit(f"max_turns must be >= 1 (got {max_turns})")
    if paths < 1:
        raise SystemExit(f"paths must be >= 1 (got {paths})")
    bad = set(axes) - set(V2_AXES)
    if bad:
        raise SystemExit(f"unknown v2 axes: {sorted(bad)} (valid: {V2_AXES})")
    scenarios: list[Any] = []
    if "upgraded" in axes:
        sweep = coverage_upgraded_scenarios(character)
        scenarios.extend(sweep[:5] if tier == "smoke" else sweep)
    if "arenas" in axes:
        scenarios.extend(_arena_cross_for_tier(tier, character))
    if "pairwise" in axes and character == "Ironclad":
        scenarios.extend(_pairwise_for_tier(tier))
    if "fuzz" in axes:
        scenarios.extend(generate_v2(s, character) for s in range(_FUZZ_SEEDS[tier]))
    # GH #500 (axis E): the auto-play/replay corner. Appended LAST on purpose —
    # the action seed is the flat position index, so anything inserted ahead of
    # an existing axis would renumber every later trial and invalidate the
    # committed Ironclad trace cache. Ironclad-only, like `pairwise`.
    if "autoplay" in axes:
        scenarios.extend(_autoplay_for_tier(tier, character))
    # GH #1021 (axis F): enchanted decks. Appended after `autoplay` for the
    # same trace-cache reason.
    if "enchant" in axes:
        from parity_fuzz_scenario import enchant_scenarios

        sweep = enchant_scenarios(character)
        scenarios.extend(sweep[:4] if tier == "smoke" else sweep)
    # item#4: replicate each scenario `paths` times with distinct action seeds.
    # The action seed is the flat position index, so paths of one scenario get
    # different seeds → different `rng.choice` play-lines, and distinct cache
    # keys (the seed is part of _trial_key). paths=1 is the original layout.
    return [(i, sc, max_turns) for i, sc in enumerate(sc for sc in scenarios for _ in range(paths))]


def _arena_cross_for_tier(tier: str, character: str = "Ironclad") -> list:
    from parity_fuzz_scenario import coverage_arena_cross_scenarios

    cross = coverage_arena_cross_scenarios(character)
    if tier == "smoke":
        # one card trial per arena — proves the plumbing end-to-end
        seen: dict[str, Any] = {}
        for sc in cross:
            assert sc.coverage_subject is not None  # arena/cross generators always set it
            act = sc.coverage_subject.rsplit("@", 1)[1]
            seen.setdefault(act, sc)
        return list(seen.values())
    if tier in ("quick", "mid"):
        return []  # full cross is a full-tier (nightly) axis
    return cross


def _drop_blocked_arenas(trials: list, bad_acts: set[str]) -> tuple[list, list[tuple[str, str]]]:
    """Remove trials whose subject's @act qualifier is in bad_acts."""
    kept, dropped = [], []
    for t in trials:
        sc = t[1]
        sub = getattr(sc, "coverage_subject", None) if sc else None
        act = sub.rsplit("@", 1)[1] if sub and "@" in sub else None
        if act in bad_acts:
            dropped.append((sub, act))
        else:
            kept.append(t)
    return kept, dropped


def _autoplay_for_tier(tier: str, character: str = "Ironclad") -> list:
    """GH #500 axis E: the auto-play/replay deck × Act-1 encounters.

    11 non-boss Act-1 encounters × N action lines. N scales with the tier
    because the deck alone does not produce the interaction — the play ORDER
    does (see ``_AUTOPLAY_LINES``): smoke 2 (22 trials, plumbing only), quick 8
    (88), mid 16 (176), full 24 (264, the calibrated size).
    """
    from parity_fuzz_scenario import autoplay_replay_scenarios

    lines = {"smoke": 2, "quick": 8, "mid": 16}.get(tier, 24)
    return autoplay_replay_scenarios(character, lines=lines)


def _pairwise_for_tier(tier: str) -> list:
    from parity_fuzz_scenario import pairwise_hook_scenarios

    sweep = pairwise_hook_scenarios()
    if tier == "smoke":
        return sweep[:3]
    return sweep  # small enough (~36 pairs) to run in full even at quick tier


def assert_wheel_fresh(rs_root: str | None = None) -> None:
    """Abort if the installed ``sts2_rs`` extension was not compiled from the
    current Rust source (issue #15 / bead ``sts2-rs-cg5ci``).

    A coverage sweep silently produces GARBAGE against a stale/wrong wheel: a
    just-wired engine change looks like it no-ops, indistinguishable from a real
    bug. Rather than infer freshness from mtimes or a git SHA (both have blind
    spots — a relinked ``.so`` over a cached rlib, or an *uncommitted* edit that
    HEAD cannot see), this is content-addressed: the ``.so`` carries
    ``src_hash()`` — a hash compiled INTO each engine crate plus the workspace
    inputs — and we demand it equals ``recompute_src_hash()`` over the on-disk
    source (the SAME Rust hasher, so the two cannot drift). With ``rs_root``
    given (wheel-venv.sh passes the resolved checkout) the source tree is taken
    from it directly and the wheel's ``direct_url.json`` absolute path is not
    consulted — so a stale or mispointed ``direct_url`` cannot matter (issue #15
    Phase 2). Otherwise the source ``crates/`` is located via ``direct_url.json``.
    Bypass with ``STS2_SKIP_WHEEL_CHECK=1``.
    """
    if os.environ.get("STS2_SKIP_WHEEL_CHECK"):
        return
    so_path = getattr(sts2_rs, "__file__", None)
    if not so_path:
        return
    pkg_dir = os.path.dirname(so_path)

    # Prefer an explicitly-passed checkout (content-addressed venvs know their
    # own source); this is what makes the wheel's direct_url.json absolute path
    # irrelevant (issue #15 Phase 2). Fall back to the editable-install record.
    crates = None
    if rs_root:
        cand = os.path.join(rs_root, "crates")
        if os.path.isdir(cand):
            crates = cand

    # Resolve the source crates/ dir from the editable-install record
    # (maturin writes url=file://<…>/crates/sts2-py). Portable: no hardcoded path.
    if not crates:
        for du in glob.glob(
            os.path.join(os.path.dirname(pkg_dir), "sts2_rs-*.dist-info", "direct_url.json")
        ):
            try:
                with open(du, encoding="utf-8") as f:
                    url = json.load(f).get("url", "")
            except (OSError, ValueError):
                continue
            if url.startswith("file://"):
                cand = os.path.dirname(url[len("file://") :].rstrip("/"))  # → crates/
                if os.path.isdir(cand):
                    crates = cand
                    break
    if not crates:
        return  # source not locatable → cannot verify, skip silently

    rs_root = os.path.dirname(crates)

    # Content-identity gate (issue #15 / bead sts2-rs-cg5ci). The loaded .so
    # carries `src_hash()` — an aggregate of a content hash compiled INTO each
    # engine crate (so a stale dependency rlib carries its OLD hash) plus the
    # workspace inputs (Cargo.lock + toolchain). Recompute the SAME hash over the
    # on-disk source via the extension's own `recompute_src_hash` (ONE hasher, in
    # Rust — no Python re-implementation to drift) and demand they match. This
    # SUPERSEDES the old mtime / per-crate .rmeta / build_git / source-version
    # heuristics: it catches a stale rlib AND uncommitted edits (which build_git,
    # keyed on HEAD, could never see), and is immune to mtime/relink races.
    baked = getattr(sts2_rs, "src_hash", None)
    recompute = getattr(sts2_rs, "recompute_src_hash", None)
    if baked is None or recompute is None:
        # Installed wheel predates the src_hash stamp. If the SOURCE emits it
        # (this change is in the tree), the wheel is stale — built before the
        # stamp and reinstalled over the current tree; otherwise it's a genuinely
        # old / CI wheel with no stamp to compare against → skip.
        librs = os.path.join(crates, "sts2-py", "src", "lib.rs")
        try:
            with open(librs, encoding="utf-8") as f:
                source_stamps = "STS2_PY_SRC_HASH" in f.read()
        except OSError:
            source_stamps = False
        if source_stamps:
            sys.exit(
                "STALE sts2_rs WHEEL (pre-src_hash) — the source stamps a content "
                "hash (sts2_rs.src_hash) but the installed extension has none, so "
                "it predates the stamp and was reinstalled over the current tree.\n"
                f"Rebuild first:  scripts/wheel-venv.sh build {rs_root}\n"
                "Bypass:         STS2_SKIP_WHEEL_CHECK=1"
            )
        return

    try:
        current = recompute(rs_root)
        stamped = baked()
    except Exception as exc:
        sys.exit(
            f"sts2_rs WHEEL PROBE FAILED ({type(exc).__name__}: {exc}) — the "
            "installed extension cannot recompute its source hash.\n"
            f"Rebuild first:  scripts/wheel-venv.sh build {rs_root}"
        )
    if stamped != current:
        sys.exit(
            "STALE sts2_rs WHEEL — engine source-hash mismatch.\n"
            f"  loaded .so src_hash = {stamped}\n"
            f"  on-disk source hash = {current}\n"
            "The installed extension was NOT compiled from the current source: a "
            "stale dependency rlib, an uncommitted edit that was not rebuilt, or a "
            "wheel from a different checkout. Content-addressed, so this catches "
            "what the old mtime / build_git gates missed.\n"
            f"Rebuild first:  scripts/wheel-venv.sh build {rs_root}\n"
            "Bypass:         STS2_SKIP_WHEEL_CHECK=1"
        )


PROBE_UPGRADE_DECK = ("STRIKE_IRONCLAD+1",) * 5
PROBE_UPGRADE_DAMAGE = 9  # STRIKE_IRONCLAD upgraded (base 6)
PROBE_ENCOUNTER = (
    "SHRINKER_BEETLE_WEAK"  # single-monster; HP rolls 38-40, irrelevant: we measure a damage DELTA
)


def _probe_damages() -> tuple[int, int]:
    """Play STRIKE_IRONCLAD+1 turn-1 on both engines; return (cs, rs) damage."""
    # All-identical deck: index-0 play is unambiguous on both engines (Rust
    # _apply_rust's hand-index binding included), and a 5-card deck fills the
    # whole opening hand.
    sc = Scenario(
        character="Ironclad",
        deck=PROBE_UPGRADE_DECK,
        hp=70,
        max_hp=80,
        relics=(),
        potions=(),
        encounter=PROBE_ENCOUNTER,
    )
    g = Game()
    try:
        cs_raw = _csharp_init(g, sc)
        cs_hp0 = cs_raw["monsters"][0]["hp"]
        _apply_csharp(g, {"action": "play_card", "card_index": 0, "target_index": 0})
        cs_dmg = cs_hp0 - g.get_state()["monsters"][0]["hp"]
        cs_monsters = [m["kind"] for m in (cs_raw.get("monsters") or [])]
    finally:
        g.close()
    rs = _rust_init(sc, monsters=cs_monsters)
    _drain_rust_choices(rs)
    rs_hp0 = normalize_rust(rs)["monsters"][0]["hp"]
    _apply_rust(rs, {"action": "play_card", "card_index": 0, "target_index": 0})
    rs_dmg = rs_hp0 - normalize_rust(rs)["monsters"][0]["hp"]
    return cs_dmg, rs_dmg


def assert_upgrade_plumbing() -> None:
    """Abort unless BOTH engines honor the "+1" deck marker.

    Plays STRIKE_IRONCLAD+1 turn-1 on each engine and demands the UPGRADED
    damage (9, base 6) from each side independently. Without this gate a
    marker-parsing regression on either side (or both!) makes every
    upgraded-sweep trial silently compare base-vs-base behavior — which
    still diffs "clean" and is indistinguishable from a real green sweep.
    Bypass: STS2_SKIP_UPGRADE_PROBE=1 (debugging only).
    """
    if os.environ.get("STS2_SKIP_UPGRADE_PROBE"):
        return
    try:
        cs_dmg, rs_dmg = _probe_damages()
    except Exception as exc:
        sys.exit(
            f"FATAL: upgrade-plumbing probe errored ({type(exc).__name__}: {exc}) — "
            "could not measure either engine. Check the C# headless builds "
            "(`just build-cs`) and the Rust wheel (`just build-rust`).\n"
            "Bypass: STS2_SKIP_UPGRADE_PROBE=1"
        )
    if cs_dmg != PROBE_UPGRADE_DAMAGE or rs_dmg != PROBE_UPGRADE_DAMAGE:
        sys.exit(
            f"FATAL: upgrade-plumbing probe failed — STRIKE_IRONCLAD+1 dealt "
            f"cs={cs_dmg} rs={rs_dmg}, expected {PROBE_UPGRADE_DAMAGE} on both. "
            f"(base STRIKE deals 6 — a 6 here means that side ignored the +1 "
            f"marker; rebuild that backend: `just build-cs` / `just build-rust`.)\n"
            "Bypass: STS2_SKIP_UPGRADE_PROBE=1"
        )


def _canon_card(cid: str | None) -> str | None:
    """Canonical card token shared by both engines.

    C# reports ``CARD.BARRICADE`` (bare model id); Rust reports the
    power-disambiguated ``BARRICADE_CARD``. Strip the C# ``CARD.`` prefix
    and the Rust ``_CARD`` suffix so the two forms collapse to one token —
    otherwise every loadable ``_CARD`` card spuriously diverges on
    hand/draw/discard ordering. Within the card namespace the bare and
    ``_CARD`` forms never name two distinct cards, so this masks nothing.

    ``_STATUS`` is the same split with one member (C# ``Sloth`` /
    ``CARD.SLOTH`` vs Rust ``SLOTH_STATUS``) and needs the same strip, or a
    Knowledge Demon board would spuriously diverge on pile ordering. Latent
    today — ``SLOTH`` is in no scenario's card list — but the asymmetry is
    the bug, not its current reach. At most ONE suffix comes off: a registry
    id carries exactly one.
    """
    if cid is None:
        return None
    if cid.startswith("CARD."):
        cid = cid[5:]
    for suffix in ("_CARD", "_STATUS"):
        if cid.endswith(suffix):
            return cid[: -len(suffix)]
    return cid


def _cs_ids(state: dict, key: str) -> list[str | None]:
    return [_canon_card(c) for c in (state.get(key) or [])]


def _rs_ids(d: dict, key: str) -> list[str | None]:
    return [_canon_card(c.get("id")) for c in (d.get("deck", {}).get(key) or [])]


def _enrich_csharp(raw: dict) -> dict:
    base = normalize_csharp(raw)
    base["hand_ids"] = _cs_ids(raw, "hand_ids")
    base["draw_ids"] = _cs_ids(raw, "draw_ids")
    base["discard_ids"] = _cs_ids(raw, "discard_ids")
    # sts2-rs item#1: exhaust pile fate + per-hand-card resolved cost were
    # emitted by BOTH engines (RunSimulator.GetStateImpl `exhaust_ids` /
    # `hand_costs`; state_dict.rs `exhaust_pile` / per-card `cost_override`)
    # but never wired into the parity diff — only hand/draw/discard *ids*
    # were compared. Comparing them catches wrong-pile routing
    # (discard-vs-exhaust) and cost-reduction / Confused drift that the
    # count-and-id-only surface missed.
    base["exhaust_ids"] = _cs_ids(raw, "exhaust_ids")
    base["hand_costs"] = [int(c) for c in (raw.get("hand_costs") or [])]
    return base


def _enrich_rust(adapter: Any) -> dict:
    base = normalize_rust(adapter)
    d = sts2_rs.state_to_dict(adapter.inner)
    base["hand_ids"] = _rs_ids(d, "hand")
    base["draw_ids"] = _rs_ids(d, "draw_pile")
    base["discard_ids"] = _rs_ids(d, "discard_pile")
    base["exhaust_ids"] = _rs_ids(d, "exhaust_pile")
    base["hand_costs"] = [
        int(c.get("cost_override", 0)) for c in (d.get("deck", {}).get("hand") or [])
    ]
    return base


def _cs_turn_sig(state: dict) -> tuple:
    """A coarse C#-state signature for detecting a non-advancing `end_turn`.

    A real turn transition (enemy turn + next player draw) always changes at
    least one of these: the three player piles, the player's HP/block/energy,
    or each monster's HP/block. If an `end_turn` leaves ALL of them identical
    while combat is still in progress, the C#-headless oracle stalled (see the
    `DoEndTurn` RetainHand carve-out in `_run_trial_core`)."""
    player = state.get("player") or {}
    return (
        tuple(state.get("hand_ids") or []),
        tuple(state.get("draw_ids") or []),
        tuple(state.get("discard_ids") or []),
        player.get("hp"),
        player.get("block"),
        player.get("energy"),
        tuple((m.get("hp"), m.get("block")) for m in (state.get("monsters") or [])),
    )


def _powers_in(state: dict) -> set[str]:
    """item#2: every power kind present on the player or any monster in a
    normalized snapshot — the corpus-coverage signal for "did this power kind
    ever actually appear in play"."""
    out: set[str] = set()
    for p in (state.get("player") or {}).get("powers") or []:
        if p.get("kind"):
            out.add(p["kind"])
    for m in state.get("monsters") or []:
        for p in m.get("powers") or []:
            if p.get("kind"):
                out.add(p["kind"])
    return out


#: item#8: structural-invariant gate. OFF by default — the synthetic coverage
#: builder (`build_scenario_state`) does NOT maintain every engine invariant the
#: real `start_combat` path guarantees, so on synthetic states the engine's own
#: suite fires on pre-existing migration debt (the d03 player/creature block
#: mirror is internal-only — the wire surface reconciles it, so C# parity stays
#: clean) and on instanced powers that the IsInstanced design intends to stack
#: (TheBomb / TagTeam / Automation). Forcing those red would drown the diff
#: signal. Enable with `--invariants` for a deliberate corruption hunt, and ON
#: for the real-vs-real full-run harness where start_combat builds the state.
_CHECK_INVARIANTS = False


def _rs_invariant_violation(adapter: Any, ctx: str) -> str | None:
    """item#8: run the Rust engine's structural invariant suite on the current
    UNCLAMPED internal state, returning the first violation message or None.

    ``state_to_dict`` clamps hp/block/energy on emit, so a differential diff
    cannot see an internal corruption (negative hp, a card duplicated across
    piles, a dangling power index). This asserts the same invariant suite the
    libfuzzer ``combat_step`` target uses, catching a shared-wrong-behavior bug
    even when C# and Rust agree on the (clamped) wire surface. No-op unless
    ``_CHECK_INVARIANTS`` is set (see its note) or the binding predates the
    ``check_invariants`` method (older wheel) — degrades to a no-op rather than
    crashing the sweep.
    """
    if not _CHECK_INVARIANTS:
        return None
    fn = getattr(adapter.inner, "check_invariants", None)
    if fn is None:
        return None
    return fn(ctx)


# Combat-LOCAL RNG streams: consumed during combat itself, so the C#
# real run and the synthetic Rust scenario-builder both start them at 0
# when the combat begins and they align read-for-read (verified on clean
# encounter trials). EXCLUDED: `up_front` and `combat_potion_generation`
# are RUN-SETUP streams (Neow / map-gen / potion-reward rolls) that the
# coverage harness's synthetic Rust state never replicates — C# carries a
# large constant baseline (e.g. up_front=410) that can never align here,
# so comparing them is pure noise. In the full-run parity_fuzz harness
# (real-vs-real from one seed) ALL streams align and this allowlist would
# widen; this set is specific to the synthetic coverage sweep.
_COMBAT_LOCAL_RNG_STREAMS = (
    "niche",
    "monster_ai",
    "shuffle",
    "combat_card_generation",
    "combat_card_selection",
    "combat_energy_costs",
    "combat_targets",
    "combat_orbs",
)

#: Coverage subjects whose effect generates a potion off the
#: ``Rng.CombatPotionGeneration`` RUN-SETUP stream (excluded above as
#: unreplicable in the synthetic coverage state). C# and Rust therefore
#: generate DIFFERENT potions and diverge on the generated potion's
#: effect/slots — a harness limitation, not an engine bug. Trials for
#: these subjects are marked clean (logged via the
#: ``combat_potion_generation_artifact`` first_path). Bead sts2-rs-s8at3.
#: - ALCHEMIZE: `CreateRandomPotionInCombat` (Alchemize.cs)
#: - ENTROPIC_BREW: `CreateRandomPotionOutOfCombat` ×3 (EntropicBrew.cs)
_POTION_GEN_ARTIFACT_SUBJECTS = frozenset(
    {
        "card:ALCHEMIZE",
        "potion:ENTROPIC_BREW",
    }
)


def _rng_desync(cs_raw: dict, rs_adapter: Any) -> list[dict]:
    """Per-stream RNG cursor mismatch between the two dumps (combat-local).

    Both engines emit ``rng_counters`` (per-stream read-count) in their
    state dump. For combat-local streams the counters align read-for-read
    in the coverage harness, so a mismatch is the exact stream + read-count
    where the engines desynced — the root pointer the value-only field diff
    cannot give when the desync is masked by a constant (a fixed-HP monster,
    a forced opener). Kept OUT of ``diff()``'s equality predicate; surfaced
    as a side-channel diagnostic.
    """
    cs = cs_raw.get("rng_counters") or {}
    rs = (sts2_rs.state_to_dict(rs_adapter.inner).get("rng_counters")) or {}
    out = []
    for k in _COMBAT_LOCAL_RNG_STREAMS:
        if cs.get(k) != rs.get(k):
            out.append({"stream": k, "csharp": cs.get(k), "rust": rs.get(k)})
    return out


def _is_victory_teardown(cs: dict, rs: dict) -> bool:
    """A2: detect a post-victory snapshot-timing artifact.

    When the played action is lethal to the last live enemy, C#'s
    ``get_state()`` already reflects post-victory teardown — monsters
    removed, draw/hand/discard piles zeroed — while the Rust harness is
    still paused in PlayerPlayPhase with its piles populated (Rust keeps
    the piles on the Victory transition; it does not zero them). ``diff()``
    then fires on a pile-count field (``discard_count`` is alphabetically
    first) and would mis-bucket the trial as PILE_COUNT, a real-bug
    category.

    We treat the trial as a teardown artifact only when BOTH engines agree
    every enemy is dead and C# has zeroed its piles. Requiring Rust to also
    show no live enemies guards against masking a genuine "Rust failed to
    register the kill" bug — in that case Rust would still have a live
    monster and this returns False, leaving normal classification to run.
    """
    cs_no_live = not any(m.get("alive") for m in (cs.get("monsters") or []))
    rs_no_live = not any(m.get("alive") for m in (rs.get("monsters") or []))
    cs_piles_zero = (
        cs.get("hand_count", 0) == 0
        and cs.get("draw_count", 0) == 0
        and cs.get("discard_count", 0) == 0
    )
    return cs_no_live and rs_no_live and cs_piles_zero


#: Fields that C# zeroes/flips on the Victory/Defeat transition (piles
#: emptied, block/powers/energy cleared, terminal flags set) but that Rust
#: leaves populated because it pauses in PlayerPlayPhase and does not tear
#: down on the terminal transition. When at least one engine has ended
#: combat, a diff on any of these is post-terminal snapshot-lifecycle skew,
#: not an engine disagreement about the game outcome. The *substantive*
#: outcome fields (player.hp, monsters[*].*) are deliberately NOT in this
#: set, so a genuine "wrong damage / kill not registered" divergence still
#: surfaces in its real class (A1/A2 harness-hygiene, 2026-05-29).
_TEARDOWN_EXACT = frozenset(
    {
        "combat_terminal",
        "player_won",
        "hand_count",
        "draw_count",
        "discard_count",
        "exhaust_count",
        "player.block",
        "player.energy",
    }
)
_TEARDOWN_PREFIX = (
    "hand_ids",
    "draw_ids",
    "discard_ids",
    "exhaust_ids",
    "hand_costs",
    "player.powers",
)


def _is_teardown_field(path: str) -> bool:
    """True if ``path`` is post-terminal teardown plumbing (see above)."""
    return path in _TEARDOWN_EXACT or path.startswith(_TEARDOWN_PREFIX)


#: Pile membership/order fields the C#-headless async in-flight pile-move
#: artifact can perturb. ``WaitForActionExecutor`` only waits for the
#: ActionExecutor to go idle, not for pending ``CardPileCmd.Draw/Add``
#: continuations to drain, so C#'s snapshot can be taken mid-transfer: a card
#: still counted in its source pile (or not yet in its destination). Rust
#: resolves pile moves synchronously within the action, so its snapshot is
#: post-transfer and CORRECT (documented in MEMORY.md
#: reference_rng_read_tracer). Bead ``sts2-rs-xk9yo``.
_INFLIGHT_PILE_FIELDS = (
    "draw_count",
    "draw_ids",
    "discard_count",
    "discard_ids",
    "hand_count",
    "hand_ids",
    # A card mid-transfer into/out of hand changes the positional hand-cost
    # list too (length + entries), so the in-flight artifact must tolerate a
    # hand_costs diff alongside hand_ids. Exhaust is deliberately EXCLUDED
    # (see _is_inflight_pile_move docstring: an exhaust diff may be a real
    # wrong-pile routing bug and must stay visible).
    "hand_costs",
)


def _is_inflight_pile_move(cs: dict, rs: dict, divs: list) -> bool:
    """True iff ``divs`` is the C#-headless async in-flight pile-move artifact.

    Conservative fingerprint (Rust correct, mark clean):
      * EVERY divergence is confined to the draw/hand/discard count/id fields
        (NOT exhaust — an exhaust diff may be a real wrong-pile routing bug;
        NOT monsters/player.hp/block/energy/powers — a vitals diff is a real
        engine bug, e.g. the mis-bucketed seeds 1338/1743);
      * the COMBINED multiset of draw+hand+discard card ids is identical
        between C# and Rust (same cards, exactly one mid-transfer between
        those piles — a dup/loss bug would break conservation and stay
        visible);
      * neither the DRAW nor the DISCARD pile differs by ORDER ALONE. GH #500:
        without this clause the fingerprint also matched a pure REORDER of the
        discard pile — same cards, different sequence — which cannot be a
        mid-transfer (nothing left the pile) and is a real divergence. It is the
        signature of a card reaching its result pile at the wrong point inside an
        awaited auto-play, and since the caller RETURNS on an artifact the trial
        was scored clean AND cut short at the first symptom: on the
        auto-play/replay deck (axis E) the suppressor swallowed 4 of 11.
        HAND order is deliberately still tolerated — that is where the genuine
        positional churn lives (it is why ``hand_costs`` is in the field list at
        all), and tightening the hand as well turned 33 more trials of the
        standard 500-trial fuzz axis red with no triage behind them.
    The caller additionally requires the RNG streams to be fully in sync.
    """
    if not divs:
        return False
    for d in divs:
        if not d["path"].startswith(_INFLIGHT_PILE_FIELDS):
            return False

    keys = ("draw_ids", "hand_ids", "discard_ids")
    cs_p = {k: list(cs.get(k) or []) for k in keys}
    rs_p = {k: list(rs.get(k) or []) for k in keys}
    if sorted(itertools.chain.from_iterable(cs_p.values())) != sorted(
        itertools.chain.from_iterable(rs_p.values())
    ):
        return False  # conservation broken -> a dup/loss bug, keep it visible
    for k in ("draw_ids", "discard_ids"):
        if cs_p[k] != rs_p[k] and sorted(cs_p[k]) == sorted(rs_p[k]):
            return False  # order-only: nothing is in flight, this is real
    return True


#: Cards whose OnPlay generates a DIFFERENT card (a status token) into a pile,
#: grounded in the C# decomp (`OnPlay` → `AddGeneratedCardToCombat(CreateCard<X>)`).
#: When such a card is caught MID-PLAY in C#'s snapshot (still in `play_ids`), Rust
#: has already spawned the token while C# has not — so the token is a legitimate
#: in-flight surplus in Rust, exactly like the self-copy case (ANGER) the carve-out
#: already allows. Keyed/valued by canonical UPPER_SNAKE id (post-`_canon_card`).
#: Tight by construction: only the SPECIFIC token each generator mints is allowed,
#: so a phantom-generation Rust bug (a card neither in-flight nor its known token)
#: still surfaces. Mirrors C# CollisionCourse/Overclock/BoostAway/FightThrough/
#: GunkUp/Turbo OnPlay (each ends with a `Cmd.Wait`, which is what lets the headless
#: snapshot land before the generation drains). Regent fuzz seed 266 (CollisionCourse
#: → Debris: C# play_ids=[COLLISION_COURSE], Debris not yet in hand; Rust resolved).
_INFLIGHT_GENERATES: dict[str, frozenset[str]] = {
    "COLLISION_COURSE": frozenset({"DEBRIS"}),
    "OVERCLOCK": frozenset({"BURN"}),
    "BOOST_AWAY": frozenset({"DAZED"}),
    "FIGHT_THROUGH": frozenset({"WOUND"}),
    "GUNK_UP": frozenset({"SLIMED"}),
    "TURBO": frozenset({"VOID"}),
}


def _is_inflight_play_pile(cs_raw: dict, rs_state: Any) -> bool:
    """True iff C#'s recorded snapshot was taken MID-PLAY — the just-played
    card is still in C#'s play pile, so its move to discard|exhaust AND its own
    queued damage/debuff/block are all still pending, while Rust (which resolves
    a play synchronously and never snapshots with a populated play pile) has
    already settled it. RUST IS CORRECT — mark clean.

    The C# headless ``WaitForActionExecutor`` waits only for the ActionExecutor
    to go idle, not for the post-``OnPlay`` ``CardPileCmd`` that moves the played
    card out of the play pile (nor the card's own queued effects) to drain. For a
    0-cost card the snapshot can therefore land before ANY of it applies — C#
    shows the card with no damage/power and not yet in discard, while Rust shows
    the fully-resolved board (the SLICE / SUPPRESS PILE_COUNT bucket of the Silent
    sweep). This is the same async-snapshot family as ``_is_inflight_pile_move``,
    but there the divergence is confined to draw/hand/discard counts (a
    ``Draw``/``Add`` continuation mid-transfer); here the in-flight card's OWN
    effect (monster hp, applied power, player block) is also unflushed, so the
    divergence set is wider. The discriminator is precise either way: a SETTLED
    C# snapshot always has an empty play pile, so a non-empty ``play_ids`` proves
    resolution is incomplete.

    Conservative fingerprint:
      * C# ``play_ids`` is non-empty (resolution provably incomplete);
      * every card is conserved — the combined multiset across ALL C# piles
        (hand/draw/discard/exhaust/play) equals Rust's across
        hand/draw/discard/exhaust — so the only difference is exactly the
        in-flight card(s) not yet flushed, never a dup/loss.
    The caller additionally requires the RNG streams to be fully in sync.
    """
    cs_play = _cs_ids(cs_raw, "play_ids")
    if not cs_play:
        return False
    cs_bag = Counter(
        _cs_ids(cs_raw, "hand_ids")
        + _cs_ids(cs_raw, "draw_ids")
        + _cs_ids(cs_raw, "discard_ids")
        + _cs_ids(cs_raw, "exhaust_ids")
        + cs_play
    )
    rd = sts2_rs.state_to_dict(rs_state.inner).get("deck", {})
    rs_bag = Counter(
        _canon_card(c.get("id"))
        for pile in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
        for c in (rd.get(pile) or [])
    )
    if cs_bag == rs_bag:
        return True
    # A mid-play card that GENERATES copies (ANGER → +1 ANGER to discard,
    # BATTLE_TRANCE-likes, etc.) breaks strict conservation: Rust has already
    # spawned the copy while the still-suspended C# snapshot has not. C# can
    # NEVER hold more of a card than Rust here (it is the one mid-resolution),
    # so require (a) C# is a sub-multiset of Rust — no card C# has is missing
    # from Rust — and (b) every surplus Rust card is a copy of a card currently
    # in C#'s `play_ids` (the in-flight generator). That keeps the fingerprint
    # tight: the only allowed discrepancy is extra copies of the exact card
    # C# is still resolving, which is provably its own unflushed generation —
    # not an arbitrary card dup/loss. Ironclad fuzz seed 64 (ANGER, discard
    # 2-vs-4) is the canonical case.
    if cs_bag - rs_bag:  # C# holds a card Rust lacks → real loss, not in-flight
        return False
    surplus = rs_bag - cs_bag
    # Allowed in-flight surplus: copies of the in-flight card itself (ANGER-style
    # self-generation) PLUS the specific status token a still-suspended generator
    # mints (CollisionCourse → Debris). Both are provably the in-flight card's own
    # unflushed generation, not an arbitrary dup/loss.
    allowed = set(cs_play)
    for card in cs_play:
        if card:
            allowed |= _INFLIGHT_GENERATES.get(card, frozenset())
    return bool(surplus) and all(card in allowed for card in surplus)


def _is_csharp_play_not_drained(
    action: dict, cs: dict, rs: dict, cs_pre_sig: tuple, rs_pre_sig: tuple
) -> bool:
    """True iff C# accepted a ``play_card`` it offered as legal but executed
    NOTHING, while Rust resolved it — the C#-headless async play-drain artifact.

    C#'s ``DoPlayCard`` (``RunSimulator.cs``) enqueues the play then calls
    ``WaitForActionExecutor``, which pumps only until the ActionExecutor reports
    idle for a short streak — NOT until the played card's async ``OnPlay``
    (``GainBlock`` + ``PowerCmd.Apply`` continuations) drains. When those
    continuations are still pending, the card never leaves C#'s hand;
    ``DoPlayCard`` hits its own ``"Card could not be played (still in hand after
    action)"`` guard and returns an Error, which the harness records as an
    unchanged (pre-play) snapshot rather than re-raising. The trial then diffs
    C#'s frozen pre-play state against Rust's fully-resolved board and mislabels
    it a Rust pile/vitals divergence (Ironclad fuzz seed 265: EQUILIBRIUM, C#
    energy 3/block 0/card-in-hand vs Rust energy 1/block 13/RetainHand applied).

    Fingerprint (Rust correct, mark clean):
      * the action is a ``play_card`` (both engines offered it — the caller's
        ``cs_actions == rs_actions`` gate already holds, and the action is chosen
        from that shared legal set);
      * C#'s post-action turn signature is BYTE-IDENTICAL to its pre-action one
        (``_cs_turn_sig`` covers all three piles + player hp/block/energy + every
        monster hp/block). A successfully played card ALWAYS at least leaves the
        hand, so an unchanged signature proves C# executed nothing;
      * Rust's signature DID advance — Rust is the only engine that resolved the
        play.
    This cannot mask a real Rust bug: a genuine illegality is rejected by C#'s
    ``CanPlay`` BEFORE enqueue (a different ``"Cannot play card X: reason"`` error
    AND the action would not appear in C#'s legal set), so it never reaches here.
    """
    if action.get("action") != "play_card":
        return False
    return _cs_turn_sig(cs) == cs_pre_sig and _cs_turn_sig(rs) != rs_pre_sig


def _is_csharp_spurious_victory(cs: dict, rs: dict) -> bool:
    """True iff C# reports ``player_won`` while its OWN monster roster still holds
    a living enemy (hp>0) and Rust keeps combat in progress — a C#-headless
    spurious-terminal artifact.

    Observed at the INIT snapshot of several 3-monster Ironclad encounters: C#
    reports ``combat_terminal``/``player_won`` True with all three monsters ALIVE
    (e.g. hp 47/42/86), its hand/energy zeroed by the bogus teardown, while Rust
    correctly shows ``combat_terminal`` False with a full opening hand and the
    monster's real starting block. A real victory requires EVERY enemy dead, so
    C# claiming the player won while it ITSELF still lists a hp>0 monster is
    internally impossible — the C# headless mis-set the terminal flag and Rust is
    correct. This cannot mask a real Rust "failed to detect a win" bug: for C# to
    have legitimately won, every enemy must be dead, so a surviving hp>0 monster
    in C#'s own state proves C# did not actually win. Gated on Rust NOT being
    terminal so genuine shared/standalone victories (handled by the
    victory-teardown / terminal-skew carve-outs) are untouched.
    """
    if cs.get("player_won") is not True:
        return False
    if rs.get("combat_terminal"):
        return False
    return any((m.get("hp") or 0) > 0 for m in (cs.get("monsters") or []))


def _is_setup_kill_niche_artifact(cs: dict, rs: dict, rng_ds: list) -> bool:
    """True iff the ONLY RNG desync is the ``niche`` (per-monster HP-roll) stream
    reading FEWER values in Rust than C#, fully explained by a setup-time monster
    death under the parity-harness roster-override (RUST IS CORRECT).

    The ``niche`` stream draws exactly one value per spawned enemy (its
    unique-max-HP roll, ``CombatState.CreateCreature`` →
    ``Creature.SetUniqueMonsterHpValue(Rng.Niche)``) at combat start. A turn-1
    relic AoE — FestivePopper's ``DamageVar(9)`` ``AfterPlayerTurnStart`` — kills
    a low-rolled enemy AFTER those rolls, so C#'s post-setup ``get_state`` roster
    (which the harness feeds back into Rust as the forced monster list when the
    native roster differs, see the ``rs_native != cs_monsters`` override below)
    is SHORT one monster. Rust then spawns only the survivors and rolls one niche
    value each, so its niche counter trails C#'s by exactly the setup-kill count
    — pure roster-override bookkeeping, not an engine desync. The surviving
    monsters' state matches (this only fires in the observable-fields-agree
    branch), so the niche-counter gap is the sole, fully-explained residual.

    Fingerprint (Rust correct, mark clean):
      * the sole desynced stream is ``niche``, with C# > Rust;
      * Rust rolled exactly one niche value per monster it currently holds
        (``rs_niche == #rs_monsters``) — faithful one-roll-per-spawn; and
      * C# rolled MORE niche values than its current roster
        (``cs_niche > #cs_monsters``) — proving a post-roll setup death.

    Cannot mask a real Rust under-roll: that would leave a spawned monster with
    no HP roll (wrong HP → a MONSTER_STATE diff) or spawn fewer monsters than C#
    (a monsters.len diff) — either surfaces as an observable-state divergence,
    handled before this rng-only branch, not as a niche-only counter gap.
    """
    if len(rng_ds) != 1 or rng_ds[0].get("stream") != "niche":
        return False
    cs_n = rng_ds[0].get("csharp") or 0
    rs_n = rng_ds[0].get("rust") or 0
    if cs_n <= rs_n:
        return False
    cs_mons = len(cs.get("monsters") or [])
    rs_mons = len(rs.get("monsters") or [])
    return rs_n == rs_mons and cs_n > cs_mons


def _is_csharp_gigantification_zero_hit_artifact(cs: dict, rs: dict, divs: list) -> bool:
    """True iff the ONLY divergence is C# retaining exactly one extra
    GIGANTIFICATION power that Rust consumed on a zero-hit (X=0) attack — a C#
    headless artifact; RUST IS CORRECT per the v0.107.1 decomp.

    `AttackCommand` fires `Hook.BeforeAttack`/`AfterAttack` UNCONDITIONALLY,
    outside the `for (i=0; i<attackCount; …)` hit loop (`AttackCommand.cs`), so a
    0-hit attack (an X-cost Whirlwind/Skewer/… played at 0 energy) still runs
    them. `GigantificationPower.BeforeAttack` latches any `IsPoweredAttack()`
    player Attack (`Move && !Unpowered`) and `AfterAttack` decrements it
    (Counter 1→0). So a 0-energy Whirlwind DOES consume Gigantification per the
    decomp — which Rust does (`hooks.rs`, with an explicit "fires even at zero
    hits" comment). The C# HEADLESS `DamageCmd.Attack().Execute()` at hitCount 0
    no-ops its Before/After-attack hooks (GodotStubs/TaskHelper swallow), so the
    oracle never consumes — a headless-only divergence (issue #184). Seed 5788.

    Fingerprint (Rust correct, mark clean):
      * every diverging path is under `player.powers` (no monster/hp/block/pile
        divergence — the 0-hit attack dealt nothing, so the only effect is the
        consume); and
      * C#'s player-power MULTISET equals Rust's PLUS exactly one
        `GIGANTIFICATION` entry (Rust holds no power C# lacks).

    Cannot mask a real Rust over-consume: Rust only ever decrements
    Gigantification inside its own `AfterAttack` on a POWERED attack — exactly
    the case the decomp says to consume — so whenever Rust consumed and C# did
    not, the decomp sides with Rust. An unpowered attack never latches, so Rust
    would not consume it either; a non-attack play never reaches the hook.
    """
    if not divs:
        return False
    if any(not d.get("path", "").startswith("player.powers") for d in divs):
        return False
    cs_bag = Counter(
        (p.get("kind"), p.get("amount")) for p in (cs.get("player") or {}).get("powers") or []
    )
    rs_bag = Counter(
        (p.get("kind"), p.get("amount")) for p in (rs.get("player") or {}).get("powers") or []
    )
    if rs_bag - cs_bag:  # Rust holds a power C# lacks → not the consume artifact
        return False
    surplus = cs_bag - rs_bag
    if sum(surplus.values()) != 1:
        return False
    ((kind, _amt),) = list(surplus.keys())
    return kind == "GIGANTIFICATION"


def _is_csharp_pile_double_count(cs: dict, rs: dict) -> bool:
    """True iff C#'s snapshot counts the SAME card instance in two piles at once
    — a card straddling hand/draw/discard/exhaust mid-transfer — so C#'s total
    card count strictly exceeds Rust's. A single card instance cannot occupy two
    piles simultaneously, so the C# snapshot is internally impossible and RUST IS
    CORRECT — mark clean. This is the spurious-victory PHILOSOPHY (detect a state
    C# itself proves cannot exist) applied to the pile surface.

    Root cause (same async-snapshot family as the other carve-outs): the headless
    ``WaitForActionExecutor`` returns before a ``CardPileCmd`` move drains, so the
    moved card is momentarily counted in BOTH its source and destination pile.
    Two sibling carve-outs already cover the adjacent shapes; this fills the gap
    between them:
      * ``_is_inflight_pile_move`` handles a pure RELOCATION — a CONSERVED
        multiset, the card gone from one pile and present in another (totals
        match); the surplus shape here breaks that conservation, so it never
        fires.
      * ``_is_inflight_play_pile`` requires a non-empty ``play_ids`` (the just-
        played card caught mid-PLAY); here the played card has already left the
        play pile and the straddle is on the DRAW that follows, so ``play_ids``
        is empty and that carve-out never fires either.
    Regent fuzz seed 306 (SHINING_STRIKE): its ``OnPlay`` returns it to draw-top,
    the next draw moves it to hand, and C#'s snapshot counts it in BOTH hand and
    draw-top — C# 17 cards / 2 SHINING_STRIKE vs Rust's correct 16 / 1.

    Fingerprint (Rust correct, mark clean):
      * Rust's full-pile multiset is a SUB-multiset of C#'s — Rust holds nothing
        C# lacks, so C# is unambiguously the surplus side (any card Rust has that
        C# lacks would be a real Rust over-count and returns False);
      * the surplus (C# minus Rust) is non-empty; and
      * EVERY surplus card occupies >=2 DISTINCT C# piles, with its surplus count
        <= (distinct C# piles it occupies - 1) — bounding the excess to one
        straddle per extra pile, provably a mid-transfer instance counted twice
        rather than an arbitrary duplication.

    This cannot mask a real Rust card-LOSS bug: a card Rust wrongly dropped sits
    in exactly ONE C# pile (its real home), so the ">=2 distinct piles" clause
    fails and the divergence correctly stays. Nor a Rust over-count: that puts a
    card in ``rs_bag - cs_bag`` and returns False at the first gate.
    """
    piles = ("hand_ids", "draw_ids", "discard_ids", "exhaust_ids")
    cs_per_pile = {p: Counter(cs.get(p) or []) for p in piles}
    cs_bag: Counter = Counter()
    for counts in cs_per_pile.values():
        cs_bag += counts
    rs_bag: Counter = Counter()
    for p in piles:
        rs_bag += Counter(rs.get(p) or [])
    if rs_bag - cs_bag:  # Rust holds a card C# lacks → not a C# surplus
        return False
    surplus = cs_bag - rs_bag
    if not surplus:
        return False
    for card, extra in surplus.items():
        n_piles = sum(1 for p in piles if cs_per_pile[p].get(card))
        if n_piles < 2 or extra > n_piles - 1:
            return False
    return True


def _is_csharp_late_victory_legal(cs: dict, csharp_only: list, rust_only: list) -> bool:
    """True iff a legal-action mismatch is the C#-headless LATE-VICTORY artifact.

    Every C# monster is dead (hp<=0 and not alive ⇒ the player has really won),
    but C#'s ``RunSimulator`` has not yet run its combat-end check, so it keeps
    the player in the play phase and still offers card plays. Rust correctly
    recognizes the all-dead board and withholds those plays (offering only
    ``end_turn``), so the only difference is C# OFFERING MORE.

    A real victory requires every enemy dead; once that holds the game ends the
    combat immediately with no further player decision point, so any card play
    C# offers on an all-dead board is non-physical. Rust dropping exactly those
    plays — and adding nothing of its own — is the correct behavior, mirroring
    the ``_is_csharp_spurious_victory`` philosophy for the legal-action surface.

    Gated tightly so a genuine legal-action divergence still surfaces: every
    C#-only action must be a ``play_card`` and Rust must add NOTHING
    (``rust_only`` empty). If Rust is missing a real action C# offers in a
    live-board state, or offers something C# does not, this returns False and
    the mismatch buckets as LEGAL_ACTION.
    """
    monsters = cs.get("monsters") or []
    if not monsters:
        return False
    # Any living enemy ⇒ not a victory state ⇒ a real comparison.
    if any((m.get("hp") or 0) > 0 or m.get("alive") for m in monsters):
        return False
    if rust_only:
        return False
    return bool(csharp_only) and all(a.get("action") == "play_card" for a in csharp_only)


def _classify(first_path: str, hands_matched_in: bool, action: dict) -> str:
    if not hands_matched_in:
        return "CASCADE"
    act_card = action.get("_card_id")
    if first_path in ("hand_ids", "draw_ids", "discard_ids") or first_path.startswith(
        ("hand_ids[", "draw_ids[", "discard_ids[")
    ):
        # ordered-list divergence with hands matching going in
        if act_card in UPGRADE_CARDS and first_path.startswith("hand_ids"):
            return "UPGRADE_CHOICE"
        return "DRAW_ORDER"
    if first_path.startswith("exhaust_ids"):
        # A card landed in the wrong pile (discard-vs-exhaust routing), a
        # different exhaust order, or a wrong exhaust count (the ``.len``
        # leaf). item#1: exhaust fate is newly compared.
        return "EXHAUST_FATE"
    if first_path.startswith("hand_costs"):
        # Per-hand-card resolved energy cost drift (cost reduction, Confused,
        # X-cost), or a hand-size mismatch surfacing as ``hand_costs.len``.
        # item#1: hand cost is newly compared.
        return "HAND_COST"
    if first_path in ("draw_count", "discard_count", "exhaust_count", "hand_count"):
        return "PILE_COUNT"
    if first_path.startswith("player.hp") or first_path == "player.hp":
        return "PLAYER_HP" if action["action"] == "end_turn" else "PLAYER_STATE"
    if first_path.startswith("player."):
        return "PLAYER_STATE"
    if first_path.startswith("monsters"):
        return "MONSTER_STATE"
    if first_path.startswith("orbs"):
        # Defect orb belt divergence (kind / passive / count / order). The
        # newly-compared orb surface (orb-parity) catches a wrong channel /
        # evoke / Focus value directly, instead of only via the downstream
        # block (Frost) or monster-HP (Lightning/Dark/Glass) cascade.
        return "ORB_STATE"
    if first_path in ("combat_terminal", "player_won"):
        return "TERMINAL"
    return f"OTHER:{first_path}"


def _trial_signature(r: dict) -> str:
    """Bucket key for a trial result.

    ``clean`` trials collapse to a single bucket. A coverage trial (one with
    a ``coverage_subject``) is keyed by that subject — so a divergence is
    attributed to the single varied element (card / relic / potion /
    encounter) no matter which action tripped it. Random trials keep the
    legacy action/card attribution.
    """
    cat = r.get("category")
    if cat == "clean":
        return "clean"
    subject = r.get("coverage_subject")
    if subject:
        return f"{cat}|{subject}"
    act = r.get("action") or {}
    card = act.get("_card_id") if isinstance(act, dict) else None
    action_name = act.get("action") if isinstance(act, dict) else None
    return f"{cat}|{card or action_name or '-'}"


# ---------------------------------------------------------------------------
# C# reference-trace cache (issue #3)
# ---------------------------------------------------------------------------
#
# ``--sweep`` is IPC/process-bound on the C# *reference* engine: every trial
# spins up a .NET headless subprocess and drives it action-by-action over
# stdio, while the Rust side it diffs against runs in microseconds. During a
# fix session only Rust changes — the vendored C# DLLs are constant — yet we
# pay the full C# cost on every sweep.
#
# The C# half of a trial is a PURE FUNCTION of ``(scenario, seed, max_turns)``:
# the action sequence is C#-driven (``rng.choice(cs_actions)``) and never
# depends on Rust, which merely mirrors the same actions. So we record each
# trial's full C# reference trace — the init dump, and per step the legal-action
# list, the chosen action, and the post-action state dump (with its
# ``rng_counters``) — once, then replay Rust-only against it on every
# subsequent sweep. No .NET spawn, no stdio: the sweep drops from minutes to
# seconds.
#
# Invalidation rides the cache NAMESPACE — a content hash of what DEFINES the
# C# oracle's behavior: the IL-patched ``lib/*.dll`` game DLLs (game version +
# headless patch) and the C# harness SOURCE (``src/**/*.cs`` — the
# decision/serialization logic). Any change to either → new namespace →
# automatic full re-record, so Rust is never diffed against a stale truth (the
# oracle analogue of ``assert_wheel_fresh``'s stale-wheel gate). Trials key on
# the FULL scenario content + seed + max_turns, so the ``assemble_v2_trials``
# shuffle (which remaps seed→scenario) can never produce a false hit.
#
# We hash the SOURCE, not the built ``Sts2Headless.dll``: a .NET build is not
# byte-reproducible across checkout paths (fresh MVID per build), so hashing the
# binary would give every worktree a different namespace and defeat reuse — and,
# critically, would stop a cache RECORDED on one machine from being REPLAYED on
# another (the committed-trace bundle for cloud sessions, data/parity-cache/).
# Source bytes ARE identical across checkouts at the same commit, so the
# namespace is stable everywhere the game DLLs + harness source match. The one
# caveat: editing C# harness source without rebuilding (`just build-cs`) flips
# the namespace while the running binary lags — use ``STS2_PARITY_NOCACHE=1``
# during C# harness development (the harness already requires a rebuilt binary
# to run at all).
#
# On by default; disable for one run with ``--no-cache`` or
# ``STS2_PARITY_NOCACHE=1``.

#: Bump when the recorded-trace schema changes (forces re-record on read).
_TRACE_FORMAT = 1

#: Repo root of sts2-pycli (…/python/parity_audit.py → …/).
_PYCLI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hash_file_into(h: hashlib._Hash, path: str) -> bool:
    """Stream ``path`` into ``h``; return True iff the file was read."""
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return True
    except OSError:
        return False


def _game_version(game_version: str | None) -> str:
    """The game build to namespace / gate: ``game_version`` or the selected one."""
    if game_version:
        return sts2_client.normalize_game_version(game_version)
    return sts2_client.GAME_VERSION


def _oracle_lib_dir(base: str, version: str) -> str:
    """``<base>/lib/<version>/``, or the legacy flat ``<base>/lib/`` (v0.107.1 only)."""
    versioned = os.path.join(base, "lib", version)
    flat = os.path.join(base, "lib")
    if (
        version == sts2_client.LEGACY_GAME_VERSION
        and not os.path.isfile(os.path.join(versioned, "sts2.dll"))
        and os.path.isfile(os.path.join(flat, "sts2.dll"))
    ):
        return flat
    return versioned


def cs_version_namespace(root: str | None = None, game_version: str | None = None) -> str:
    """Content hash identifying the C# oracle version (path-independent).

    Returns ``<game version>-<16 hex>``, e.g. ``v0.107.1-6d8df59872fd67e9``.
    ``game_version`` defaults to the selected build
    (``sts2_client.GAME_VERSION`` ← ``STS2_GAME_VERSION``). The hash covers
    what determines the oracle's behavior:
      * the game version itself, so two builds can never share a namespace;
      * ``lib/<version>/*.dll`` — that build's vendored, IL-patched game DLLs
        (game version + the headless patch, since ``sts2.dll`` is shipped
        already patched);
      * ``src/**/*.cs`` / ``*.csproj`` — the C# harness source (the
        decision/serialization logic that produces the trace), EXCLUDING
        ``bin/``/``obj/`` build output.

    Deliberately content-addressed and build-INdependent: the namespace is
    identical across every checkout (worktree / fresh clone / cloud session) at
    the same commit, so a recorded trace bundle is reusable everywhere. (Hashing
    the built ``Sts2Headless.dll`` would not be — .NET embeds a fresh MVID per
    build, so identical source at two paths yields different bytes.)

    Returns the empty string when no oracle bytes can be located (so the caller
    disables the cache rather than collapsing every version into one bucket).
    """
    base = root or _PYCLI_ROOT
    version = _game_version(game_version)
    files: list[str] = []
    libdir = _oracle_lib_dir(base, version)
    if os.path.isdir(libdir):
        files += [os.path.join(libdir, n) for n in os.listdir(libdir) if n.endswith(".dll")]
    srcdir = os.path.join(base, "src")
    for dirpath, dirnames, filenames in os.walk(srcdir):
        dirnames[:] = [d for d in dirnames if d not in ("bin", "obj")]
        # ``.cs`` is the harness logic; ``.csproj`` carries runtime config that
        # changes the oracle's BEHAVIOR (the `RollForward` policy decides whether
        # the app runs on a .NET runtime where Harmony can install its patches at
        # all — on CoreCLR 10+ every patch silently no-ops, degrading the whole
        # oracle). So the project file is part of the oracle version too.
        files += [os.path.join(dirpath, fn) for fn in filenames if fn.endswith((".cs", ".csproj"))]

    h = hashlib.sha256()
    h.update(f"game_version={version}".encode())
    h.update(b"\0")
    hashed = 0
    # Sort by path-relative-to-base so the order is identical across checkouts.
    for p in sorted(files, key=lambda q: os.path.relpath(q, base)):
        rel = os.path.relpath(p, base)
        h.update(rel.encode())
        h.update(b"\0")
        if _hash_file_into(h, p):
            hashed += 1
        h.update(b"\0")
    return f"{version}-{h.hexdigest()[:16]}" if hashed else ""


def assert_oracle_fresh(root: str | None = None, game_version: str | None = None) -> None:
    """Abort if the selected build's ``Sts2Headless.dll`` is older than its C# source.

    Checks ``src/**/bin/<game version>/**/Sts2Headless.dll`` — the oracle for
    ``game_version`` (default: the selected build, ``STS2_GAME_VERSION``), so a
    fresh v0.107.1 build cannot vouch for a stale v0.111.0 one or vice versa.

    item#8 (stale-oracle gate). ``cs_version_namespace`` hashes the ``.cs``
    SOURCE — by design, so a recorded trace bundle replays across checkouts at
    the same commit. But that means a stale BUILT binary serves a namespace
    that *claims* source-freshness: edit (or ``git pull``) a ``.cs`` file
    without ``just build-cs`` and the harness happily runs the OLD binary while
    caching its traces under the NEW source hash. This bit a full session —
    commit 09641672 (#23) added ``hand_costs`` to RunSimulator.GetStateImpl, but
    the on-disk ``Sts2Headless.dll`` predated the merge, so every sweep on main
    silently compared against a pre-#23 oracle (``hand_costs`` absent) under the
    #23 namespace.

    The fix is a cheap mtime gate, symmetric with ``assert_wheel_fresh`` for the
    Rust side: if any ``src/**/*.cs`` is newer than the newest built
    ``Sts2Headless.dll``, the binary is stale → abort with the rebuild command.
    A ``git pull`` that touches C# source restamps those files newer than an old
    dll, so this also catches "pulled new oracle, forgot to rebuild". Bypass
    with ``STS2_SKIP_ORACLE_CHECK=1`` (e.g. mid C#-harness edit before rebuild).
    """
    if os.environ.get("STS2_SKIP_ORACLE_CHECK"):
        return
    base = root or _PYCLI_ROOT
    version = _game_version(game_version)
    srcdir = os.path.join(base, "src")
    if not os.path.isdir(srcdir):
        return  # source not locatable → cannot verify, skip silently

    # Newest built Sts2Headless.dll for this game version, across any build
    # configuration (bin/<version>/Debug, bin/<version>/Release).
    build_marker = os.sep + "bin" + os.sep + version + os.sep
    dll_mtime = 0.0
    dll_seen = False
    for cfg_root, _, filenames in os.walk(srcdir):
        if build_marker not in cfg_root + os.sep:
            continue
        for fn in filenames:
            if fn == "Sts2Headless.dll":
                dll_seen = True
                dll_mtime = max(dll_mtime, pathlib.Path(os.path.join(cfg_root, fn)).stat().st_mtime)

    # Newest C# source under src/, excluding build output.
    src_mtime = 0.0
    newest_src = ""
    for dirpath, dirnames, filenames in os.walk(srcdir):
        dirnames[:] = [d for d in dirnames if d not in ("bin", "obj")]
        for fn in filenames:
            if fn.endswith(".cs"):
                m = pathlib.Path(os.path.join(dirpath, fn)).stat().st_mtime
                if m > src_mtime:
                    src_mtime, newest_src = m, os.path.join(dirpath, fn)

    if not dll_seen:
        sys.exit(
            f"C# oracle NOT BUILT for {version} — no Sts2Headless.dll under src/**/bin/{version}/.\n"
            f"Build first:  just build-cs {version}\n"
            "Bypass:       STS2_SKIP_ORACLE_CHECK=1"
        )
    if src_mtime > dll_mtime:
        sys.exit(
            "STALE C# ORACLE — a .cs source file is newer than the built dll, so "
            "the running oracle predates the source the trace cache is namespaced "
            "by (it would silently compare against an OLD oracle under the NEW "
            f"namespace).\n  newest source = {os.path.relpath(newest_src, base)}\n"
            f"Rebuild first:  just build-cs {version}\n"
            "Bypass:         STS2_SKIP_ORACLE_CHECK=1"
        )


def _trial_key(seed: int, max_turns: int, sc: Scenario) -> str:
    """Stable cache key for a trial: a hash of its full identity.

    Keys on the scenario CONTENT (not just the seed), so a sweep that remaps
    seed→scenario (``assemble_v2_trials`` shuffles) can never collide with a
    differently-shaped trial that happened to reuse the same seed index.
    """
    payload = json.dumps(
        {"seed": seed, "max_turns": max_turns, "scenario": dataclasses.asdict(sc)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class _CsOracle:
    """The C# touchpoints a trial needs, in call order: ``init`` → per step
    ``legal`` / ``apply`` / ``snapshot`` → ``close``.

    Two implementations: :class:`_LiveCsOracle` drives a real headless
    subprocess (the default, no-cache path); :class:`_ReplayCsOracle` serves a
    recorded trace with no subprocess at all.
    """

    def init(self) -> dict:
        raise NotImplementedError

    def legal(self) -> list[dict]:
        raise NotImplementedError

    def apply(self, action: dict) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _LiveCsOracle(_CsOracle):
    """Drives a fresh C# headless ``Game`` — byte-for-byte the legacy path."""

    __slots__ = ("_g", "_sc")

    def __init__(self, sc: Scenario):
        self._sc = sc
        self._g = Game()

    def init(self) -> dict:
        return _csharp_init(self._g, self._sc)

    def legal(self) -> list[dict]:
        return _csharp_legal(self._g)

    def apply(self, action: dict) -> None:
        # F3 (audit 2026-05-29): drain the C# card_select choice (e.g.
        # ARMAMENTS' FromHandForUpgrade) symmetrically with Rust's, so both
        # engines are snapshotted AFTER choice resolution. Without it C#'s
        # snapshot is paused mid-OnPlay with the played card still in PlayPile,
        # spuriously reading discard_count 1 lower (PILE_COUNT|ARMAMENTS).
        res = _apply_csharp(self._g, action)
        _drain_csharp_choices(self._g, res)

    def snapshot(self) -> dict:
        return self._g.get_state()

    def close(self) -> None:
        try:
            self._g.close()
        except Exception:
            pass


class _ReplayCsOracle(_CsOracle):
    """Replays a recorded C# trace — no .NET subprocess.

    Serves the same call sequence ``_run_trial_core`` issues, advancing one
    step per :meth:`snapshot`. A C# harness error captured during recording is
    re-raised at the matching call so the replayed trial buckets identically
    (HARNESS_INIT / HARNESS_LEGAL / HARNESS_APPLY / HARNESS_POST).
    """

    __slots__ = ("_i", "_steps", "_trace")

    def __init__(self, trace: dict):
        self._trace = trace
        self._steps = trace.get("steps") or []
        self._i = 0

    def init(self) -> dict:
        err = self._trace.get("init_error")
        if err is not None:
            raise RuntimeError(err)
        return self._trace["init"]

    def legal(self) -> list[dict]:
        step = self._steps[self._i]
        err = step.get("error")
        if err and err[0] == "legal":
            raise RuntimeError(err[1])
        return step["actions"]

    def apply(self, action: dict) -> None:
        step = self._steps[self._i]
        err = step.get("error")
        if err and err[0] == "apply":
            raise RuntimeError(err[1])

    def snapshot(self) -> dict:
        step = self._steps[self._i]
        err = step.get("error")
        if err and err[0] == "post":
            raise RuntimeError(err[1])
        snap = step["snap"]
        self._i += 1
        return snap

    def close(self) -> None:
        pass


def _record_cs_trace(seed: int, max_turns: int, sc: Scenario) -> dict:
    """Drive ONLY the C# engine to completion, recording its reference trace.

    Decoupled from Rust: it does NOT apply the both-terminal short-circuit
    (which needs Rust) and records every step until C# itself runs out of
    actions or ``max_turns`` is reached. That guarantees a later replay against
    fixed Rust — which only ever stops at or BEFORE C#'s natural end — never
    out-runs the trace. Action selection mirrors ``_run_trial_core`` exactly
    (same ``random.Random(seed ^ 0xDEADBEEF)`` + ``rng.choice``), so each
    recorded action is the one replay will pick. The bare action (no
    ``_card_id`` attribution) is fed to C#, which ignores the extra key.
    """
    rng = random.Random(seed ^ 0xDEADBEEF)
    g = Game()
    steps: list[dict] = []
    trace: dict[str, Any] = {"v": _TRACE_FORMAT}
    try:
        try:
            trace["init"] = _csharp_init(g, sc)
        except Exception as exc:
            trace["init_error"] = repr(exc)
            trace["steps"] = steps
            return trace
        turn = 0
        while turn < max_turns:
            try:
                cs_actions = _csharp_legal(g)
            except Exception as exc:
                steps.append({"error": ["legal", repr(exc)]})
                break
            if not cs_actions:
                steps.append({"actions": []})  # terminal: no action to apply
                break
            action = rng.choice(cs_actions)
            try:
                res = _apply_csharp(g, action)
                _drain_csharp_choices(g, res)
            except Exception as exc:
                steps.append(
                    {"actions": cs_actions, "action": action, "error": ["apply", repr(exc)]}
                )
                break
            try:
                snap = g.get_state()
            except Exception as exc:
                steps.append(
                    {"actions": cs_actions, "action": action, "error": ["post", repr(exc)]}
                )
                break
            steps.append({"actions": cs_actions, "action": action, "snap": snap})
            if action["action"] == "end_turn":
                turn += 1
    finally:
        try:
            g.close()
        except Exception:
            pass
    trace["steps"] = steps
    return trace


class _TraceCache:
    """Filesystem cache of recorded C# reference traces, namespaced by the
    oracle version (:func:`cs_version_namespace`)."""

    __slots__ = ("_dir", "namespace")

    def __init__(self, root: str, namespace: str):
        self.namespace = namespace
        self._dir = os.path.join(root, namespace)
        os.makedirs(self._dir, exist_ok=True)

    def get_or_record(self, seed: int, max_turns: int, sc: Scenario) -> tuple[dict, str]:
        """Return ``(trace, "hit"|"miss")`` — loading the cached trace or
        recording (and persisting) a fresh one on miss."""
        path = os.path.join(self._dir, _trial_key(seed, max_turns, sc) + ".json")
        try:
            with open(path, encoding="utf-8") as f:
                trace = json.load(f)
            if trace.get("v") == _TRACE_FORMAT:
                return trace, "hit"
        except (OSError, ValueError):
            pass
        trace = _record_cs_trace(seed, max_turns, sc)
        self._write_atomic(path, trace)
        return trace, "miss"

    @staticmethod
    def _write_atomic(path: str, trace: dict) -> None:
        # Per-pid tmp + atomic rename: parallel workers each record DIFFERENT
        # keys, so there is no same-file contention, but a crash mid-write must
        # never leave a truncated trace a later run would load as truth. A
        # write failure (read-only disk) is non-fatal — the trace was already
        # returned in-memory.
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(trace, f, separators=(",", ":"))
            pathlib.Path(tmp).replace(path)
        except OSError:
            try:
                pathlib.Path(tmp).unlink()
            except OSError:
                pass


#: Process-global cache, set by :func:`_init_worker_cache` in the parent and in
#: every pool worker (via ``ProcessPoolExecutor(initializer=…)``, so it survives
#: any start method, not just fork). ``None`` ⇒ the legacy live path.
_CACHE: _TraceCache | None = None


def _init_worker_cache(
    cache_dir: str | None, namespace: str | None, check_invariants: bool = False
) -> None:
    """Pool initializer (also called in the parent): install ``_CACHE`` and the
    process-global ``_CHECK_INVARIANTS`` flag (item#8) so every worker honors
    ``--invariants``."""
    global _CACHE, _CHECK_INVARIANTS
    _CACHE = _TraceCache(cache_dir, namespace) if (cache_dir and namespace) else None
    _CHECK_INVARIANTS = check_invariants


def run_trial(seed: int, max_turns: int = 6, scenario: Scenario | None = None) -> dict:
    """Run one parity trial, replaying a cached C# trace when ``_CACHE`` is set.

    With no cache (the default) the C# engine is driven live — behaviorally
    identical to the legacy path. With a cache, the C# half is served from
    (or recorded into) the trace cache and Rust is diffed against it; the
    result carries a transient ``_cache`` status the caller tallies and pops.
    """
    # Coverage mode supplies an explicit single-variable scenario; random mode
    # synthesizes one from the seed. The action-selection RNG always keys off
    # `seed` so a trial is reproducible either way.
    sc = scenario if scenario is not None else generate(seed)
    cache = _CACHE
    if cache is None:
        return _run_trial_core(seed, max_turns, sc, _LiveCsOracle(sc))
    trace, status = cache.get_or_record(seed, max_turns, sc)
    out = _run_trial_core(seed, max_turns, sc, _ReplayCsOracle(trace))
    out["_cache"] = status
    return out


def _run_trial_core(seed: int, max_turns: int, sc: Scenario, oracle: _CsOracle) -> dict:
    """Diff Rust against the C# oracle for one trial.

    The C# side is reached only through ``oracle`` — a live headless ``Game``
    (``_LiveCsOracle``) or a replayed reference trace (``_ReplayCsOracle``).
    Rust is always live. The action sequence keys off ``seed`` (``rng.choice``
    over the C# legal actions), so it is identical under either oracle.
    """
    rng = random.Random(seed ^ 0xDEADBEEF)
    out: dict[str, Any] = {
        "seed": seed,
        "scenario": dataclasses.asdict(sc),
        "coverage_subject": sc.coverage_subject,
        "category": None,
        "first_path": None,
        "action": None,
        "turn": 0,
        "divergences": None,
        "hand_before": None,
    }
    # item#2: per-trial coverage facts. `cards_played` keeps multiplicity (for
    # the ">= N plays" gate); `powers_seen` / `monsters_seen` are presence sets.
    # Stored by reference so the populated values reach every `return out`.
    cov: dict[str, Any] = {
        "cards_played": [],
        "powers_seen": set(),
        "monsters_seen": set(),
    }
    out["_coverage"] = cov
    try:
        try:
            cs_raw = oracle.init()
            cs_monsters = [m["kind"] for m in (cs_raw.get("monsters") or [])]
            cov["monsters_seen"].update(m for m in cs_monsters if m)
            # Per-monster initial move-state id C# rolled, aligned with
            # `cs_monsters`. Force-matches RNG-branched starter rotations
            # (ScrollsOfBiting{Weak,Normal}'s `StarterMoveIdx = NextInt(3)`)
            # that the synthetic Rust scenario builder can't roll itself.
            # No-op for non-rotating monsters. Bead sts2-rs-u3wjj.
            cs_move_states = [m.get("move_state") for m in (cs_raw.get("monsters") or [])]
            # Prefer Rust's NATIVE encounter spawn when its roster already
            # matches C#'s order+kinds. The kind-only override (sts2-rs-3010t)
            # exists for RNG-branched `GenerateMonsters` rosters
            # (FLYCONID_NORMAL / SLITHERING_STRANGLER / SLIMES_*) where the
            # Rust roster would otherwise differ — but it passes ONLY kinds and
            # drops per-monster `MonsterSpawnArgs` (KinFollower `StartsWithDance`,
            # Inklet slot). For a deterministic-roster encounter like
            # THE_KIN_BOSS the override silently demotes the dancing follower to
            # a plain QUICK_SLASH opener (monsters[0].powers.len 2/1 + a player.hp
            # cascade). Build native first; only fall back to the override when
            # the native roster genuinely diverges from C# (sts2-rs-zdjjx).
            # GH #1054: with C#'s floor the native build rolls RNG-branched
            # rosters itself, so it also matches when a setup AoE killed a
            # monster `cs_monsters` (living only) no longer lists. Traces
            # recorded before the headless reported it carry no total_floor.
            floor = cs_raw.get("total_floor")
            rs_state = _rust_init(
                sc, monsters=None, move_states=cs_move_states, encounter_floor=floor
            )
            rs_native = [m.get("kind") for m in (_enrich_rust(rs_state).get("monsters") or [])]
            if rs_native != cs_monsters:
                rs_state = _rust_init(
                    sc, monsters=cs_monsters, move_states=cs_move_states, encounter_floor=floor
                )
            _drain_rust_choices(rs_state)
        except Exception as exc:
            out["category"] = "HARNESS_INIT"
            out["divergences"] = repr(exc)
            return out

        cs = _enrich_csharp(cs_raw)
        rs = _enrich_rust(rs_state)
        cov["powers_seen"].update(_powers_in(cs))
        rng_ds = _rng_desync(cs_raw, rs_state)
        inv = _rs_invariant_violation(rs_state, "init")
        if inv:
            out["category"] = "INVARIANT"
            out["first_path"] = "invariant"
            out["divergences"] = inv
            out["turn"] = 0
            return out
        divs = diff(cs, rs)
        if divs and _is_csharp_spurious_victory(cs, rs):
            # C#-headless spurious-terminal artifact: C# reports player_won at
            # the init snapshot while its own roster still holds a living enemy,
            # zeroing its hand/energy/block, while Rust correctly runs the combat.
            # RUST IS CORRECT (logged via `csharp_spurious_victory_artifact`,
            # never silent). Ironclad fuzz seeds 57/76/116/295/297/311.
            out["category"] = "clean"
            out["first_path"] = "csharp_spurious_victory_artifact"
            out["divergences"] = divs[:25]
            out["turn"] = 0
            return out
        if divs:
            out["category"] = _classify(divs[0]["path"], True, {"action": "init"})
            out["first_path"] = divs[0]["path"]
            out["divergences"] = divs[:25]
            if rng_ds:
                out["rng_desync"] = rng_ds
            return out
        if rng_ds and _is_setup_kill_niche_artifact(cs, rs, rng_ds):
            # Roster-override niche artifact (see helper): a turn-1 setup AoE
            # (FestivePopper) killed a low-rolled enemy after its HP roll, so the
            # forced cs_monsters roster is short one monster and Rust rolls one
            # fewer niche value than C#. Surviving monsters match — RUST IS
            # CORRECT. Mark clean (logged, never silent). Seeds 2418/3847.
            out["category"] = "clean"
            out["first_path"] = "setup_kill_niche_artifact"
            out["rng_desync"] = rng_ds
            out["divergences"] = rng_ds
            out["turn"] = 0
            return out
        if rng_ds:
            # Observable fields agree but an RNG stream desynced — a latent
            # divergence (manifests under other seeds / later turns). The
            # value diff can't see it; the counters can.
            out["category"] = "RNG_DESYNC"
            out["first_path"] = "rng:" + rng_ds[0]["stream"]
            out["rng_desync"] = rng_ds
            out["divergences"] = rng_ds
            return out

        # sts2-rs-jejon: once a `Rng.CombatPotionGeneration` consumer
        # actually FIRES in this trial (an ALCHEMIZE play or an
        # ENTROPIC_BREW use), the generated potion is rolled off a
        # RUN-SETUP stream the synthetic scenario never replicates, so C#
        # and Rust hold DIFFERENT potions from that action onward. Every
        # later divergence (belt contents, use_potion legal actions, the
        # generated potion's effect) is the same harness artifact the
        # coverage-subject guard below suppresses for `card:ALCHEMIZE` /
        # `potion:ENTROPIC_BREW` subjects; this flag extends the guard to
        # fuzz trials where `coverage_subject` is None. Set when the
        # chosen action is the generator itself (the same-step post-check
        # is already artifact: the rolled potion differs immediately).
        potion_gen_tainted = False
        # sts2-rs-xk9yo (jejon follow-up): also taint when the unreplicable
        # `combat_potion_generation` counter INCREASES on any step. The chosen
        # action `_card_id` check below only catches ALCHEMIZE/ENTROPIC_BREW
        # played as a TRACKED action; it misses ALCHEMIZE auto-played by
        # CASCADE's `AutoPlayFromDrawPile` (seed 3105) and any other indirect
        # CombatPotionGeneration consumer. A counter advance is a precise
        # signal that a generated potion entered the belt off the run-setup
        # stream the synthetic scenario never replicates → divergence from that
        # point on is the harness artifact, not an engine bug.
        prev_potion_gen = (cs_raw.get("rng_counters") or {}).get("combat_potion_generation", 0)
        # orb-parity: the analogue of the potion-gen taint for the Defect orb
        # belt. A RANDOM orb channel (TrashToTreasure, "channel a random orb")
        # rolls off `Rng.CombatOrbGeneration` — a stream the synthetic scenario
        # never replicates — so C# and Rust channel DIFFERENT orbs from that
        # point on, and every later orb/block/HP diff is the same harness
        # artifact. Deterministic channels (Lightning/Frost/Dark from cards) do
        # NOT read this stream, so the taint only fires on genuine random
        # generation and leaves real orb bugs visible. A counter advance is the
        # precise signal (mirrors `prev_potion_gen`).
        orb_gen_tainted = False
        prev_orb_gen = (cs_raw.get("rng_counters") or {}).get("combat_orbs", 0)
        turn = 0
        while turn < max_turns:
            # Combat-over short-circuit. When BOTH engines agree the combat
            # has ended with the SAME outcome, no further action is
            # meaningful — stop the trial cleanly. This must precede the
            # legal-action comparison: for a terminal state the two
            # harnesses disagree on a pure REPRESENTATION fallback — C#'s
            # `RunSimulator.LegalActions` emits a trivial `[end_turn]`,
            # while the Rust core returns `[]` (terminal ⇒ no actions, the
            # correct MCTS leaf-eval semantics). That is not an engine
            # divergence about the game outcome, so comparing it spuriously
            # buckets every random-play DEATH (starter deck vs a multi-enemy
            # encounter ⇒ defeat by turn ~4-5) as LEGAL_ACTION. Gating on
            # matching `player_won` keeps a genuine outcome disagreement
            # (one engine won, the other lost / only one ended) visible: it
            # falls through to the normal comparison below. Mirrors the
            # existing `_is_victory_teardown` / terminal_skew philosophy for
            # state diffs.
            if (
                cs.get("combat_terminal")
                and rs.get("combat_terminal")
                and cs.get("player_won") == rs.get("player_won")
            ):
                break
            try:
                cs_actions = oracle.legal()
                rs_actions = _rust_legal(rs_state)
            except Exception as exc:
                out["category"] = "HARNESS_LEGAL"
                out["divergences"] = repr(exc)
                return out
            if cs_actions != rs_actions:
                if sc.coverage_subject in _POTION_GEN_ARTIFACT_SUBJECTS or potion_gen_tainted:
                    # sts2-rs-s8at3: ENTROPIC_BREW fills potion slots with
                    # potions rolled off the unreplicable
                    # `Rng.CombatPotionGeneration` stream, so C# and Rust end
                    # up with different potion slots → different use_potion
                    # legal actions. Harness limitation, not an engine bug
                    # (see the post-action artifact skip below). Mark clean.
                    out["category"] = "clean"
                    out["first_path"] = "combat_potion_generation_artifact"
                    out["turn"] = turn
                    return out
                cs_only = [a for a in cs_actions if a not in rs_actions]
                rs_only = [a for a in rs_actions if a not in cs_actions]
                if _is_csharp_late_victory_legal(cs, cs_only, rs_only):
                    # C# headless kept offering card plays after its last enemy
                    # died (combat_terminal still False); Rust correctly returned
                    # only end_turn. A C#-oracle late-victory artifact, not an
                    # engine divergence. Mark clean.
                    out["category"] = "clean"
                    out["first_path"] = "terminal_skew_csharp_late_victory"
                    out["turn"] = turn
                    return out
                out["category"] = "LEGAL_ACTION"
                out["first_path"] = "legal_actions"
                out["divergences"] = {
                    "csharp_only": cs_only,
                    "rust_only": rs_only,
                }
                out["turn"] = turn
                return out
            if not cs_actions:
                break

            action = rng.choice(cs_actions)
            # attribute: what card id sits at the chosen hand index?
            hands_matched_in = cs.get("hand_ids") == rs.get("hand_ids")
            if action["action"] == "play_card":
                ci = action.get("card_index", 0)
                hand = cs.get("hand_ids") or []
                action = dict(action)
                action["_card_id"] = hand[ci] if 0 <= ci < len(hand) else None
                if action["_card_id"]:
                    # item#2: a card actually PLAYED (vs merely sitting in the
                    # deck) — the coverage signal the random-play corpus gates on.
                    cov["cards_played"].append(action["_card_id"])
            elif action["action"] == "use_potion":
                # attribute to the specific potion (slot order == scenario order)
                pi = action.get("potion_index", 0)
                action = dict(action)
                action["_card_id"] = (
                    f"POTION:{sc.potions[pi]}" if 0 <= pi < len(sc.potions) else "POTION:?"
                )

            # sts2-rs-jejon: taint the trial when a CombatPotionGeneration
            # consumer fires (see the flag's init above for the rationale).
            gen_base = (action.get("_card_id") or "").split("+")[0].removesuffix("_CARD")
            if gen_base in ("ALCHEMIZE", "POTION:ENTROPIC_BREW"):
                potion_gen_tainted = True

            # Snapshot BOTH engines' turn signatures BEFORE the action so an
            # `end_turn` that the C# headless oracle fails to advance (see the
            # stall carve-out below) can be detected by a C# byte-identical
            # post-state while Rust advanced.
            rs_pre_sig = _cs_turn_sig(rs)
            cs_pre_sig = _cs_turn_sig(cs)
            cs_pre_turn = cs.get("turn")
            try:
                # oracle.apply drains the C# card_select choice symmetrically
                # with Rust's, so both engines are snapshotted AFTER choice
                # resolution (see _LiveCsOracle.apply for the F3 rationale).
                oracle.apply(action)
                _apply_rust(rs_state, action)
                _drain_rust_choices(rs_state)
            except Exception as exc:
                out["category"] = "HARNESS_APPLY"
                out["action"] = action
                out["divergences"] = repr(exc)
                return out

            try:
                cs_raw_snap = oracle.snapshot()
                cs = _enrich_csharp(cs_raw_snap)
                rs = _enrich_rust(rs_state)
                cov["powers_seen"].update(_powers_in(cs))
            except Exception as exc:
                out["category"] = "HARNESS_POST"
                out["divergences"] = repr(exc)
                return out

            # item#8: a structural invariant violation (negative hp, a card
            # duplicated/lost across piles, a dangling power index) is an engine
            # bug regardless of C# parity, and the wire surface clamps it away —
            # so check it unconditionally, BEFORE any potion-taint/terminal
            # carve-out could mark the trial clean.
            inv = _rs_invariant_violation(rs_state, action.get("action", "?"))
            if inv:
                out["category"] = "INVARIANT"
                out["first_path"] = "invariant"
                out["action"] = action
                out["turn"] = turn
                out["divergences"] = inv
                return out

            rng_ds = _rng_desync(cs_raw_snap, rs_state)
            # sts2-rs-xk9yo: taint when the unreplicable CombatPotionGeneration
            # stream advanced this step (a generated potion entered the belt) —
            # see `prev_potion_gen` init above. Done BEFORE the divergence
            # classification so this step's potion/belt divergence is suppressed.
            cur_potion_gen = (cs_raw_snap.get("rng_counters") or {}).get(
                "combat_potion_generation", 0
            )
            if cur_potion_gen > prev_potion_gen:
                potion_gen_tainted = True
            prev_potion_gen = cur_potion_gen
            # orb-parity: taint when the unreplicable `combat_orbs` stream
            # advanced this step (a random orb entered the belt). Done BEFORE the
            # divergence classification so this step's orb/block/HP diff is
            # suppressed — see `prev_orb_gen` init above.
            cur_orb_gen = (cs_raw_snap.get("rng_counters") or {}).get("combat_orbs", 0)
            if cur_orb_gen > prev_orb_gen:
                orb_gen_tainted = True
            prev_orb_gen = cur_orb_gen
            divs = diff(cs, rs)
            if (
                divs
                and action.get("action") == "end_turn"
                and not cs.get("combat_terminal")
                and cs.get("turn") == cs_pre_turn
                and _cs_turn_sig(rs) != rs_pre_sig
            ):
                # C#-headless `RunSimulator.DoEndTurn` ORACLE STALL: under the
                # headless SuppressYield/sync-context model, the enemy turn can
                # fail to run to completion — `PlayerCmd.EndTurn` deadlocks on a
                # RetainHand power (STABLE_SERUM, EQUILIBRIUM), on some boss
                # boards, or after an ASYNC enemy continuation (a Decimillipede
                # segment's `ReattachMove` heal — `Cmd.Wait` mid-`ExecuteEnemyTurn`
                # — drops the surviving segments' attacks). The stuck-retry path
                # `UndoReadyToEndTurn`s back to the SAME play-phase round.
                #
                # The robust discriminator is the C# ROUND COUNTER: a real
                # `end_turn` ALWAYS advances it (player turn -> enemy turn -> next
                # player turn = round N+1). If C#'s `turn` is unchanged AND C# did
                # not end combat, the turn provably did not complete — C# is stuck
                # — while Rust reached a conclusion (its turn signature changed:
                # advanced the round, or the enemy killed the player). Crucially,
                # if C# had COMPLETED the turn (survived to round N+1) this would
                # be a real comparison and `turn` would differ — so a frozen C#
                # round is unambiguous and cannot mask a Rust bug (we simply
                # cannot compare a half-resolved C# turn; Rust is the only
                # fully-resolved party).
                #
                # This supersedes the old full-`_cs_turn_sig`-frozen test, which
                # missed PARTIAL stalls where C# advances SOME state (a segment
                # reattach heals 0->25) but never finishes the turn — Necrobinder
                # fuzz seed 384 (Decimillipede: C# player takes 0 dmg / survives,
                # Rust correctly dies) and seed 8 (C# stuck at hand=1/discard=8,
                # Rust draws a fresh hand). Logged via
                # `csharp_endturn_oracle_stall_artifact` so the drop is never
                # silent.
                out["category"] = "clean"
                out["first_path"] = "csharp_endturn_oracle_stall_artifact"
                out["turn"] = turn
                out["divergences"] = divs[:25]
                return out
            if divs and orb_gen_tainted:
                # A random orb was generated off `Rng.CombatOrbGeneration`; C#
                # and Rust hold different orbs, so the resulting orb/block/HP
                # divergence is a harness artifact (mirrors the potion-gen
                # carve-out below). Logged via the `combat_orb_generation_artifact`
                # first_path, never silent.
                out["category"] = "clean"
                out["first_path"] = "combat_orb_generation_artifact"
                out["turn"] = turn
                return out
            if divs and (
                sc.coverage_subject in _POTION_GEN_ARTIFACT_SUBJECTS or potion_gen_tainted
            ):
                # sts2-rs-s8at3: ALCHEMIZE (CreateRandomPotionInCombat) and
                # ENTROPIC_BREW (CreateRandomPotionOutOfCombat) both roll the
                # generated potion off `Rng.CombatPotionGeneration` — a
                # RUN-SETUP stream the synthetic coverage state never
                # replicates (same exclusion as the rng-desync allowlist
                # above). C# and Rust therefore generate DIFFERENT potions and
                # diverge on the generated potion's effect/slots. This is a
                # harness limitation, not an engine divergence — mark clean,
                # mirroring the terminal/teardown artifact handling. Logged so
                # the drop is never silent.
                out["category"] = "clean"
                out["first_path"] = "combat_potion_generation_artifact"
                return out
            if divs and _is_csharp_spurious_victory(cs, rs):
                # C#-headless spurious-terminal artifact (see helper): C# reports
                # player_won with a living enemy still on its roster while Rust
                # runs the combat. RUST IS CORRECT. Mark clean (logged via the
                # `csharp_spurious_victory_artifact` first_path, never silent).
                out["category"] = "clean"
                out["first_path"] = "csharp_spurious_victory_artifact"
                out["action"] = action
                out["turn"] = turn
                out["divergences"] = divs[:25]
                return out
            if divs and _is_csharp_play_not_drained(action, cs, rs, cs_pre_sig, rs_pre_sig):
                # C#-headless async play-drain artifact: C# accepted a play_card
                # it offered as legal but its WaitForActionExecutor returned
                # before the card's OnPlay drained, so DoPlayCard's "still in hand
                # after action" guard left C# frozen at its pre-play state while
                # Rust resolved the play. C# turn-sig byte-identical to pre-action
                # + Rust advanced ⇒ RUST IS CORRECT. Mark clean (logged via the
                # `csharp_play_not_drained_artifact` first_path, never silent).
                # Ironclad fuzz seed 265 (EQUILIBRIUM).
                out["category"] = "clean"
                out["first_path"] = "csharp_play_not_drained_artifact"
                out["action"] = action
                out["turn"] = turn
                out["divergences"] = divs[:25]
                return out
            if divs and _is_inflight_play_pile(cs_raw_snap, rs_state):
                # C#-headless async in-flight PLAY-pile artifact: C#'s snapshot
                # was taken before the just-played card drained out of the play
                # pile, so its damage/debuff/block and its move to discard are
                # all still pending while Rust has fully resolved it. C#
                # `play_ids` non-empty + every card conserved ⇒ C# is provably
                # MID-PLAY and RUST IS CORRECT. Mark clean (the SLICE / SUPPRESS
                # PILE_COUNT bucket of the Silent sweep), mirroring the
                # inflight-pile-move carve-out (logged via the
                # `inflight_play_pile_artifact` first_path, never silent).
                #
                # Unlike the pile-move carve-out, this one does NOT require
                # `not rng_ds`: a mid-play multi-hit / targeted attack (WHIRLWIND,
                # FEAR, ANGER) advances the `combat_targets` (and kindred
                # play-driven) RNG stream in Rust as part of resolving the card,
                # but NOT yet in the still-suspended C# snapshot. That desync is
                # the SAME in-flight card's unflushed effect observed on the RNG
                # stream instead of the board — exactly what `play_ids`
                # non-empty + full card conservation already prove. C# cannot be
                # meaningfully compared while it is half-resolved, so deferring to
                # Rust is the only sound call (the rng_ds is recorded for audit).
                # Ironclad fuzz seed 275 (WHIRLWIND, combat_targets 1-vs-2) is the
                # canonical case this previously mis-bucketed as PILE_COUNT.
                out["category"] = "clean"
                out["first_path"] = "inflight_play_pile_artifact"
                out["action"] = action
                out["turn"] = turn
                out["divergences"] = divs[:25]
                if rng_ds:
                    out["rng_desync"] = rng_ds
                return out
            if divs and not rng_ds and _is_inflight_pile_move(cs, rs, divs):
                # C#-headless async in-flight pile-move artifact: a still-pending
                # CardPileCmd.Draw/Add was not drained before C#'s snapshot, so
                # C# counts a card in its source pile while Rust (synchronous
                # pile moves) already moved it. RNG fully in sync + identical
                # draw/hand/discard multiset (one card mid-transfer) + no
                # monster/player-vitals diff ⇒ RUST IS CORRECT. Mark clean,
                # mirroring the terminal/teardown/potion-gen artifact carve-outs
                # (logged, never silent). Bead sts2-rs-xk9yo.
                out["category"] = "clean"
                out["first_path"] = "inflight_pile_move_artifact"
                out["action"] = action
                out["turn"] = turn
                out["divergences"] = divs[:25]
                return out
            if divs and _is_csharp_pile_double_count(cs, rs):
                # C#-headless async pile DOUBLE-COUNT artifact: a CardPileCmd move
                # (here a draw following a card's OnPlay-return-to-draw) was caught
                # before it drained, so C# counts the same card instance in two
                # piles at once — its total card count strictly exceeds Rust's. A
                # single instance cannot occupy two piles, so C#'s snapshot is
                # internally impossible and RUST IS CORRECT (the spurious-victory
                # philosophy on the pile surface). Distinct from the conserved-
                # multiset relocation (`_is_inflight_pile_move`) and the mid-PLAY
                # pile (`_is_inflight_play_pile`, which needs a non-empty
                # play_ids). Mark clean (logged via the
                # `csharp_pile_double_count_artifact` first_path, never silent).
                # Regent fuzz seed 306 (SHINING_STRIKE in BOTH hand and draw-top:
                # C# 17 cards / 2 SHINING_STRIKE vs Rust 16 / 1).
                out["category"] = "clean"
                out["first_path"] = "csharp_pile_double_count_artifact"
                out["action"] = action
                out["turn"] = turn
                out["divergences"] = divs[:25]
                return out
            if divs and _is_csharp_gigantification_zero_hit_artifact(cs, rs, divs):
                # C#-headless zero-hit Gigantification artifact (see helper): a
                # 0-energy X-cost attack (Whirlwind) consumes Gigantification per
                # the decomp (AfterAttack fires even at hitCount 0), which Rust
                # does; the C# headless no-ops the 0-hit attack's hooks and keeps
                # the power. RUST IS CORRECT. Mark clean (logged, never silent).
                # Seed 5788.
                out["category"] = "clean"
                out["first_path"] = "csharp_gigantification_zero_hit_artifact"
                out["action"] = action
                out["turn"] = turn
                out["divergences"] = divs[:25]
                return out
            if divs:
                paths = [d["path"] for d in divs]
                if (
                    cs.get("combat_terminal")
                    and rs.get("combat_terminal")
                    and cs.get("player_won") == rs.get("player_won")
                ):
                    # A3: BOTH engines ended combat on THIS action with the
                    # SAME outcome (the common case is a shared DEFEAT to an
                    # end-of-turn Constrict/Poison tick in the
                    # SLITHERING_STRANGLER arena around turn 6). Because they
                    # AGREE on combat_terminal + player_won, neither flag
                    # appears in the diff paths, so the A1 branch below never
                    # fires; the only residual diff is board state on an
                    # already-decided combat — a surviving enemy's uncleared
                    # block (cs=0 rs=5), monster hp, or piles. C# advances one
                    # extra sub-step (the enemy turn-start `AfterTurnStart`
                    # block clear) before registering the loss, while the Rust
                    # core short-circuits the end_turn chain on the lethal
                    # end-of-player-turn tick (combat_loop.rs end_turn_dispatch
                    # returns CombatLost from end_player_turn_phase_two, before
                    # enemy_turn_start). The game OUTCOME is identical, so this
                    # is a terminal-snapshot representation artifact, symmetric
                    # to the both-VICTORY `_is_victory_teardown` carve-out
                    # (A2) and the terminal-disagreement carve-out (A1).
                    # Verified via trace_subject.py for AKABEKO / WHIRLWIND /
                    # BAG_OF_MARBLES / STONE_ARMOR (sts2-rs-lmfsm): all four
                    # share player_hp 8->0 at end_player_turn_phase_two with
                    # the lone monsters[*].block 0-vs-5 residual.
                    out["category"] = "clean"
                    out["first_path"] = "terminal_skew_both_defeat"
                    out["action"] = action
                    out["turn"] = turn
                    out["divergences"] = divs[:25]
                    return out
                if _is_victory_teardown(cs, rs):
                    # A2: both engines killed the last enemy; the only
                    # remaining diffs are C#'s post-victory pile/relic
                    # teardown vs Rust's still-populated piles. Outcome
                    # agrees → clean, not an engine divergence.
                    out["category"] = "clean"
                    out["first_path"] = "terminal_teardown"
                elif "combat_terminal" in paths or "player_won" in paths:
                    # A1: at least one engine has ended combat. Strip the
                    # teardown plumbing (terminal flags + piles/block/powers
                    # C# zeroes on the transition) and re-classify on the
                    # first substantive field (hp / monster state). If every
                    # substantive field agrees, the diff is pure terminal
                    # snapshot-lifecycle skew → clean; otherwise the real
                    # divergence (e.g. monsters[1].hp) surfaces in its class.
                    substantive = [d for d in divs if not _is_teardown_field(d["path"])]
                    if not substantive:
                        out["category"] = "clean"
                        out["first_path"] = "terminal_skew"
                    else:
                        out["category"] = _classify(
                            substantive[0]["path"], hands_matched_in, action
                        )
                        out["first_path"] = substantive[0]["path"]
                else:
                    out["category"] = _classify(divs[0]["path"], hands_matched_in, action)
                    out["first_path"] = divs[0]["path"]
                out["action"] = action
                out["turn"] = turn
                out["divergences"] = divs[:25]
                out["hand_before"] = {
                    "csharp": (cs.get("hand_ids")),
                }
                if rng_ds:
                    out["rng_desync"] = rng_ds
                return out
            if rng_ds and potion_gen_tainted:
                # sts2-rs-jejon: the differing generated potions consume
                # different stream reads when used — same artifact, not a
                # latent engine desync. Mark clean (logged, never silent).
                out["category"] = "clean"
                out["first_path"] = "combat_potion_generation_artifact"
                out["action"] = action
                out["turn"] = turn
                return out
            if rng_ds:
                # Observable fields agree but an RNG stream desynced this
                # turn — latent divergence; surface it as its own bucket.
                out["category"] = "RNG_DESYNC"
                out["first_path"] = "rng:" + rng_ds[0]["stream"]
                out["rng_desync"] = rng_ds
                out["divergences"] = rng_ds
                out["action"] = action
                out["turn"] = turn
                return out
            if action["action"] == "end_turn":
                turn += 1
    finally:
        oracle.close()
    out["category"] = "clean"
    return out


def _run_one(item: tuple[int, Scenario | None, int]) -> dict:
    """Process-pool worker: run a single trial and return its result dict.

    Top-level (picklable) so it can be dispatched to a
    ``ProcessPoolExecutor``. Each call builds its own C# ``Game`` harness +
    Rust state inside ``run_trial`` and tears them down, so trials are fully
    independent and order-insensitive (each is seeded by its own index).
    A crash is captured into a ``CRASH:<type>`` result rather than killing
    the pool.
    """
    seed, sc, max_turns = item
    try:
        r = run_trial(seed=seed, max_turns=max_turns, scenario=sc)
        # item#2: the coverage sets are not picklable-friendly for the manifest
        # / examples.json; freeze them to sorted lists for the parent to merge.
        cov = r.get("_coverage")
        if cov:
            cov["powers_seen"] = sorted(cov["powers_seen"])
            cov["monsters_seen"] = sorted(cov["monsters_seen"])
        return r
    except BaseException as ex:
        return {
            "seed": seed,
            "coverage_subject": getattr(sc, "coverage_subject", None),
            "category": f"CRASH:{type(ex).__name__}",
            "first_path": None,
            "divergences": repr(ex),
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=300)
    p.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help=(
            "Turns per trial. For the v2 sweep this is the deep-horizon control "
            "(item#3): default 6 reproduces the original sweep; raise it (e.g. "
            "30) to reach late-combat scaling bugs. Part of the trace-cache key, "
            "so a deeper run records its own traces."
        ),
    )
    p.add_argument(
        "--paths",
        type=int,
        default=1,
        help=(
            "item#4 (multi-path): independent random action trajectories per v2 "
            "scenario. >1 explores the SAME scenario down N different play-lines "
            "(distinct action seeds), surfacing combos one random path misses. "
            "Most useful with the fuzz axis."
        ),
    )
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--out", default="/tmp/parity_audit")
    p.add_argument(
        "--coverage",
        action="store_true",
        help=(
            "Run the exhaustive single-variable coverage sweep "
            "(every card / relic / potion / encounter exactly once) instead "
            "of the random seed sweep. --trials is ignored; buckets are keyed "
            "by the varied element."
        ),
    )
    p.add_argument(
        "--coverage-character",
        default="Ironclad",
        choices=["Ironclad", "Silent", "Defect", "Regent", "Necrobinder", "Fool"],
        help=(
            "Which character's coverage pack to sweep with --coverage. "
            "'Ironclad' (default) is the full IC+colorless+relic+potion+"
            "encounter sweep; 'Silent'/'Defect'/'Necrobinder' mirror it with "
            "their class cards; 'Regent' is the class-specific surface (88 "
            "Regent cards + Regent relic pool). Defect/Regent/Necrobinder were "
            "ported from the Python sim and never A/B'd against C#."
        ),
    )
    p.add_argument(
        "--coverage-subjects",
        default=None,
        help=(
            "Regex; with --coverage, keep only trials whose coverage_subject "
            "matches. Lets a sweep target one slice of content instead of the "
            "whole single-variable pack -- e.g. "
            "--coverage-subjects 'encounter:(TOADPOLES|SEAPUNK|CORPSE_SLUGS)' "
            "for the Underdocks act (GH #544). The action seed is still the "
            "index in the FILTERED list, so a filtered run is reproducible on "
            "its own terms but its trials are not the same play-lines the "
            "unfiltered sweep would give them."
        ),
    )
    p.add_argument(
        "--sweep-character",
        default="Ironclad",
        choices=["Ironclad", "Silent", "Defect", "Regent", "Necrobinder", "Fool"],
        help=(
            "Which character's card pool drives the v2 (--sweep v2) random-seed "
            "axes (upgraded / arenas / fuzz). Default 'Ironclad' reproduces the "
            "original sweep exactly. The pairwise axis is Ironclad-only and is "
            "skipped for other characters."
        ),
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help=(
            "Number of parallel worker processes. 0 (default) = os.cpu_count(). "
            "1 = serial (easiest for debugging / pdb). Each worker spins up its "
            "own C# headless harness, so peak RAM scales with this; lower it if "
            "the box thrashes."
        ),
    )
    p.add_argument("--sweep", choices=["v1", "v2"], default="v1")
    p.add_argument(
        "--invariants",
        action="store_true",
        help=(
            "item#8: after every action, run the engine's structural invariant "
            "suite (HP/block/energy bounds, PowerStore integrity, deck-partition "
            "conservation, hand cap, enemy mirror) on the UNCLAMPED Rust state "
            "and fail the trial on a violation. OFF by default — the synthetic "
            "coverage builder doesn't maintain every invariant real start_combat "
            "does, so this is a deliberate corruption hunt, not a default gate."
        ),
    )
    p.add_argument("--tier", choices=["smoke", "quick", "mid", "full"], default="quick")
    p.add_argument(
        "--coverage-report",
        action="store_true",
        help=(
            "item#2: after the sweep, write coverage_report.json (cards PLAYED "
            "with counts, powers/monsters SEEN) and print the pool elements the "
            "corpus never exercised — so 'card sat in deck but was never played' "
            "stops being a silent blind spot."
        ),
    )
    p.add_argument(
        "--coverage-min",
        type=int,
        default=0,
        help=(
            "item#2: fail (exit 2) if any card in the sweep character's pool was "
            "PLAYED fewer than N times across the corpus. 0 (default) = report "
            "only, no gate. Implies --coverage-report."
        ),
    )
    p.add_argument(
        "--axes",
        default="upgraded,arenas,fuzz,pairwise,autoplay,enchant",
        help="comma-separated v2 axes subset",
    )
    p.add_argument(
        "--manifest-dir",
        default=os.path.normpath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "sts2-rs", "docs", "audits", "sweeps"
            )
        ),
        help="where v2 manifests are written + compared",
    )
    p.add_argument(
        "--compare-to",
        default=None,
        help="explicit previous manifest path (default: latest same-tier)",
    )
    p.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cache the C# reference traces (issue #3), ON BY DEFAULT. The first "
            "sweep records each trial's full C# trace; subsequent sweeps skip "
            "the .NET subprocess entirely and replay Rust-only against the "
            "cached truth. Namespaced by a content hash of the C# oracle "
            "(lib/*.dll + src/**/*.cs), so any game/harness change "
            "auto-invalidates. Use --no-cache (or STS2_PARITY_NOCACHE=1) to "
            "force a fully-live sweep."
        ),
    )
    p.add_argument(
        "--cache-dir",
        default=os.environ.get("STS2_PARITY_CACHE_DIR")
        or os.path.join(os.path.expanduser("~"), ".cache", "sts2_parity"),
        help=(
            "Directory for the C# trace cache (default: $STS2_PARITY_CACHE_DIR "
            "or ~/.cache/sts2_parity). Traces live under a per-oracle-version "
            "subdir."
        ),
    )
    sts2_client.add_game_version_arg(p)
    args = p.parse_args(argv)
    sts2_client.apply_game_version_arg(args)

    # C# reference-trace cache (issue #3). Compute the oracle-version namespace
    # ONCE in the parent, then install `_CACHE` here (serial path) and in every
    # pool worker via the ProcessPoolExecutor initializer below — so it survives
    # any start method, not just fork. STS2_PARITY_NOCACHE=1 fully disables it.
    cache_dir: str | None = None
    namespace: str | None = None
    if args.cache and not os.environ.get("STS2_PARITY_NOCACHE"):
        namespace = cs_version_namespace()
        if namespace:
            cache_dir = args.cache_dir
            print(
                f"[cache] enabled — C# {namespace} @ {os.path.join(args.cache_dir, namespace)}",
                file=sys.stderr,
            )
        else:
            print(
                "[cache] DISABLED — could not locate C# oracle DLLs (lib/*.dll, "
                "Sts2Headless.dll) to namespace; running live.",
                file=sys.stderr,
            )
    _init_worker_cache(cache_dir, namespace, args.invariants)

    # Never sweep against a stale engine — a pre-edit wheel makes a real
    # fix look like a no-op (and vice-versa). Cheap mtime gate; bypass via
    # STS2_SKIP_WHEEL_CHECK=1. Runs ONCE in the parent before any worker
    # forks, so the gate isn't paid per-process.
    assert_wheel_fresh()
    # Symmetric gate for the C# side (item#8): the trace-cache namespace hashes
    # .cs SOURCE, so a stale built dll silently serves an old oracle under a
    # fresh namespace. Abort if the binary lags its source. Bypass via
    # STS2_SKIP_ORACLE_CHECK=1.
    assert_oracle_fresh()

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out, ts)
    os.makedirs(out_dir, exist_ok=True)

    blocked: dict[str, str] = {}
    axes: list[str] = []
    if args.sweep == "v2":
        axes = [a for a in args.axes.split(",") if a]
        if {"upgraded", "fuzz", "pairwise"} & set(axes):
            assert_upgrade_plumbing()
        trials = assemble_v2_trials(
            args.tier,
            axes,
            args.sweep_character,
            max_turns=args.max_turns,
            paths=args.paths,
        )
        print(
            f"[sweep-v2:{args.tier}:{args.sweep_character}] {len(trials)} trials, axes={axes}",
            file=sys.stderr,
        )
    elif args.coverage:
        sweep = coverage_scenarios(args.coverage_character)
        if args.coverage_subjects:
            pat = re.compile(args.coverage_subjects)
            kept = [sc for sc in sweep if sc.coverage_subject and pat.search(sc.coverage_subject)]
            if not kept:
                raise SystemExit(
                    f"--coverage-subjects {args.coverage_subjects!r} matched none of "
                    f"the {len(sweep)} {args.coverage_character} coverage subjects"
                )
            print(
                f"[coverage:filter] {len(kept)}/{len(sweep)} subjects match "
                f"{args.coverage_subjects!r}",
                file=sys.stderr,
            )
            sweep = kept
        # deterministic per-trial action seed = index, so reruns reproduce
        trials = [(i, sc, args.max_turns) for i, sc in enumerate(sweep)]
        print(
            f"[coverage:{args.coverage_character}] {len(trials)} single-variable trials",
            file=sys.stderr,
        )
    else:
        trials = [(args.seed_start + i, None, args.max_turns) for i in range(args.trials)]

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    # No point spawning more workers than trials.
    jobs = max(1, min(jobs, len(trials)))
    print(f"[parallel] {jobs} worker process(es)", file=sys.stderr)

    buckets: Counter[str] = Counter()
    # keep one representative trial per signature (category + varied element)
    examples: dict[str, dict] = {}
    results: list[dict] = []
    cache_tally: Counter[str] = Counter()  # "hit"/"miss" when caching is on
    # item#2: corpus coverage accumulators.
    cards_played: Counter[str] = Counter()
    powers_seen: set[str] = set()
    monsters_seen: set[str] = set()

    def _record(r: dict) -> None:
        # The transient cache status never belongs in the persisted result /
        # manifest — pop it for the tally and move on.
        status = r.pop("_cache", None)
        if status:
            cache_tally[status] += 1
        # item#2: drain + merge this trial's coverage facts (also keeps the
        # transient set out of the persisted result / manifest).
        cov = r.pop("_coverage", None)
        if cov:
            # cards_played entries come from `hand_ids`, already canonicalized.
            cards_played.update(cov.get("cards_played", []))
            powers_seen.update(cov.get("powers_seen", []))
            monsters_seen.update(cov.get("monsters_seen", []))
        cat = r.get("category")
        sig = _trial_signature(r)
        buckets[sig] += 1
        if cat not in ("clean",) and sig not in examples:
            examples[sig] = r
        results.append(r)
        label = r.get("coverage_subject") or r.get("seed")
        print(f"[{label}] {sig} :: {r.get('first_path')}", file=sys.stderr)

    # Pre-flight: check each non-act1 arena against a plain starter deck
    # BEFORE running the cross, so a dirty arena blocks its whole column
    # instead of generating ~600 cascade-noise divergences.
    if args.sweep == "v2" and "arenas" in axes and args.tier != "quick":
        from parity_fuzz_scenario import arena_preflight_scenarios

        bad_acts: set[str] = set()
        for i, sc in enumerate(arena_preflight_scenarios()):
            assert sc.coverage_subject is not None  # arena/cross generators always set it
            # seed-offset past the pool trials so log lines stay unambiguous
            r = _run_one((len(trials) + i, sc, _V2_MAX_TURNS))
            _record(r)
            if r.get("category") != "clean":
                bad_acts.add(sc.coverage_subject.split(":", 1)[1])
        if bad_acts:
            trials, dropped = _drop_blocked_arenas(trials, bad_acts)
            for sub, act in dropped:
                blocked[sub] = f"arena {act} pre-flight not clean"
            print(
                f"[arena-preflight] BLOCKED acts: {sorted(bad_acts)} "
                f"({len(dropped)} trials skipped)",
                file=sys.stderr,
            )
            print(
                f"[sweep-v2:{args.tier}] {len(trials)} trials after arena-preflight drop",
                file=sys.stderr,
            )

    if jobs == 1:
        for item in trials:
            _record(_run_one(item))
    else:
        # chunksize batches tasks per worker to amortize IPC; the C# harness
        # startup per trial dominates, so a small chunk keeps the pool busy
        # without starving late workers. The initializer installs `_CACHE` in
        # every worker regardless of start method (fork/spawn/forkserver).
        with _cf.ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_init_worker_cache,
            initargs=(cache_dir, namespace, args.invariants),
        ) as pool:
            for r in pool.map(_run_one, trials, chunksize=1):
                _record(r)

    with open(os.path.join(out_dir, "buckets.json"), "w", encoding="utf-8") as f:
        json.dump(dict(buckets.most_common()), f, indent=2)
    with open(os.path.join(out_dir, "examples.json"), "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, default=str)
    print(f"\n=== {out_dir} ===", file=sys.stderr)
    for k, v in buckets.most_common():
        print(f"{v:5d}  {k}", file=sys.stderr)

    if namespace:
        hits, misses = cache_tally["hit"], cache_tally["miss"]
        total = hits + misses
        print(
            f"[cache] served {hits}/{total} from cache (C# {namespace}); recorded {misses}",
            file=sys.stderr,
        )

    if args.sweep == "v2":
        # imported late: keeps parity_audit importable even if sweep_manifest
        # grows deps; also avoids any future circular-import risk
        import sweep_manifest as sm

        try:
            pycli_sha = (
                subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                ).stdout.strip()
                or "unknown"
            )
        except OSError:
            pycli_sha = "unknown"
        meta = {
            "date": ts,
            "sts2_rs_sha": getattr(sts2_rs, "build_git", lambda: "unknown")(),
            "pycli_sha": pycli_sha,
            "seed_range": [0, len(trials)],
        }
        # Normalize defensively: a raw `..`-containing dir makes
        # latest_manifest's path-equality exclusion of the just-written
        # manifest fail, so the run would diff against itself (always zero
        # regressions). normpath collapses `..` so both writer and reader agree.
        manifest_dir = os.path.normpath(args.manifest_dir)
        manifest = sm.build_manifest(args.tier, axes, results, meta, blocked=blocked)
        path = sm.write_manifest(manifest, manifest_dir, args.tier)
        prev_path = args.compare_to or sm.latest_manifest(manifest_dir, args.tier, before=path)
        print(f"[manifest] {path}", file=sys.stderr)
        if prev_path:
            with open(prev_path, encoding="utf-8") as f:
                d = sm.compare_manifests(json.load(f), manifest)
            for k in ("regressions", "fixes", "new", "gone"):
                if d[k]:
                    print(f"[{k}] {len(d[k])}: {', '.join(d[k][:10])}", file=sys.stderr)
            if d["regressions"]:
                return 1

    # item#2: corpus coverage report + optional gate. Measures what the sweep
    # ACTUALLY exercised (cards played, powers/monsters seen) vs the character's
    # full pool — turning "card sat in the deck but was never played" from a
    # silent blind spot into a reported (optionally failing) fact.
    if args.coverage_report or args.coverage_min:
        from parity_fuzz_scenario import _v2_card_pool

        pool = sorted({cc for c in _v2_card_pool(args.sweep_character) if (cc := _canon_card(c))})
        unplayed = [c for c in pool if cards_played.get(c, 0) == 0]
        under = sorted(
            (c for c in pool if 0 < cards_played.get(c, 0) < (args.coverage_min or 1)),
            key=lambda c: cards_played[c],
        )
        report = {
            "character": args.sweep_character,
            "tier": args.tier,
            "pool_size": len(pool),
            "cards_played_distinct": sum(1 for c in pool if cards_played.get(c, 0)),
            "cards_unplayed": unplayed,
            "coverage_min": args.coverage_min,
            "cards_under_min": {c: cards_played[c] for c in under},
            "play_counts": dict(cards_played.most_common()),
            "powers_seen": sorted(powers_seen),
            "monsters_seen": sorted(monsters_seen),
        }
        with open(os.path.join(out_dir, "coverage_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(
            f"[coverage] {report['cards_played_distinct']}/{len(pool)} pool cards "
            f"played; {len(unplayed)} never played; "
            f"{len(powers_seen)} power kinds + {len(monsters_seen)} monster kinds seen",
            file=sys.stderr,
        )
        if unplayed:
            print(f"[coverage] UNPLAYED: {', '.join(unplayed[:40])}", file=sys.stderr)
        if args.coverage_min and (unplayed or under):
            short = unplayed + under
            print(
                f"[coverage] GATE FAIL: {len(short)} card(s) played < "
                f"{args.coverage_min}× (need more seeds or targeted scenarios)",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
