"""The campaign environment (WP2): one Ironclad A10 v0.111.0 run, every stored decision in order.

Plan `docs/plans/2026-11-02-ironclad-e2e-ppo-v0111.md`: §3.2 (decisions), §3.3 (practice
segments, see `practice.py`), §6 (reward, and what a stored row means), §7 (teacher-acting
units). PREREG-0 (`docs/experiments/2026-11-03-ironclad-e2e-v111/PREREG.md`) fixes the
reward and the caps.

`E2EEnv` walks one segment: a whole run from Neow, or a practice segment that starts at a
captured map decision. It yields every *stored* decision, combat and out-of-combat (OOC),
in order, and returns an `E2ETerminal` at the end::

    env = E2EEnv(seed)
    d = env.reset()
    while isinstance(d, E2EDecision):
        d = env.step(actor.choose(d))  # a merged-class index, see `E2EDecision.members`

**Stored decisions.** A step is stored iff it has at least two merged classes (Appendix A of
the plan). A step with one legal action or one merged class is applied here with no forward
pass and no row (auto-advance), but the fight's `IntentLog` still sees every step that can
change it (encode.py:59-66): the log only records on the first step of a round, so a forced
step later in a round skips the snapshot, which nothing would read (`FAST_PATHS`).

**Classification and dedupe are the schema's** (`ppo.e2e.contracts`, SCHEMA.md §5-§6):
`classify_combat` / `classify_ooc` give the decision type (`E2EDecision.dtype`, a
`contracts.DT` id) and an action kind per member, from the action set, never from
`phase()` (pending deck operations and potion swaps preempt the phase). A kind outside its
type raises `contracts.SchemaError`, which ends the segment as an error and counts
`unknown`. The event is bound (`materialize_event`) before its options are read
(#300/#302). Members merge by `contracts.combat_class_keys` (from the belief view) and
`contracts.ooc_class_keys`; the representative is a class's first member. Never by
`play_edge_key` (snapshot.py:487-503).

Deviation, recorded: SCHEMA.md §5 rule T6 says a played Rampage and a fresh one stay
apart because the played one has a higher `base_damage`. On W1 both read
`base_damage` 10; the played copy's +5 is only in `combat_vars.rampage_damage`, which
`_card()` drops, and `card_damage_preview()` shows 10 for both (the played one deals 15).
So `contracts.card_key` merges them. The env refines each play key by the per-copy face
variables in `FACE_VARS`, which keeps T6. The refinement only splits classes, so every
schema invariant on members (byte-identical tokens) still holds.

**Rows and reward** (plan §6, PREREG-0): win +1.0; +0.05 per boss beaten (act 1, act 2,
act-3 boss 1; the boss whose win ends the run earns only the win); +0.002 per new floor;
death or a cap 0; no HP shaping. `rewards[t]` is all reward on the transition from stored
row t to the next stored row (or the end), auto-advanced steps included, which is the
`rew[t]` of `review/scripts/gae_two_chain.py`. Each decision also carries `reward`, the
amount just assigned to the previous row, for streaming consumers. Reward that accrues
before a segment's first stored row (a practice segment's entry floor) belongs to no row
and is counted in `unassigned_reward`. So `sum(rewards) == total_reward` (T4).

On v0.111.0 the act-3 bosses are not back to back in one combat phase: boss 1 (floor 45)
routes to `MapChoice` with one forced Move, then boss 2 (floor 46). A fight therefore ends
when the phase leaves combat; runslot.py's round-reset check (:287-292) is kept as a
fallback and counted in `counts["round_reset"]` (expected 0).

**Caps** (plan §3.2): 100 turns or 1,000 stored decisions per fight, 6,000 stored
decisions per run; a capped segment ends as a loss (`E2ETerminal.cap`), counted for the
>0.1% alarm. A safety cap on all steps (forced ones included) backs them up. The
per-decision caps (255 members, 64 merged classes) are asserts in SCHEMA.md §7, so a
breach ends the segment as an error and counts `assert/members` or `assert/classes`
(`check_caps`).

**Teacher-acting units** (plan §7, U0): every decision names its `unit`, `("fight", k)`
for combat rows and `("act", a)` for OOC rows, and `unit_start` marks the unit's first
stored row, so the actor draws beta_act once per fight and once per act.

**The schema's env decision** (SCHEMA.md §8.1): `E2EDecision.decision` is the stored decision
as a `contracts.EnvDecision`, which `contracts.validate_decision` accepts and the tokenizer
encodes. `E2EDecision` itself keeps `actions` as the live `cs.legal_actions()` in combat,
which the teachers read (wp/4b-net); the `EnvDecision`'s `actions` are the members in the
belief view's index space. Either way the actor acts with `E2EEnv.step(c)`, which applies
the live representative of class `c`. A `contracts.Tokenizer` passed in `EnvConfig` is
driven as §8.2 says: `reset_fight()` at every fight start and `observe(policy)` at every
combat step, forced ones included, before the action; the actor calls `encode`.

**F9** (plan §3.5): every step, forced ones included, goes through
`InvariantChecker.maybe_check` (`ppo.e2e.invariants`), which checks 1% of them from its own
RNG. Violations are counted (`f9/violations`, `f9/<code>`) and the first few kept in
`E2EEnv.f9_violations`, which the actor sends with the segment's end record; one is a
lane stop (plan §8) unless the segment ended at the step cap (a hang: PREREG Amendment 9,
`learner._f9_hung`), so the caller reads the counts.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import itertools
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sts2_client

from ppo.e2e import contracts as C
from ppo.e2e import factory
from ppo.e2e.invariants import SAMPLE_RATE, InvariantChecker
from ppo.env import (
    CombatTurn,
    PpoRunEnv,
    Terminal,
    assert_encodable,
    classify_surface,
)
from ppo.env import Decision as RunDecision

# ── the campaign run (plan §3.1) ─────────────────────────────────────────────

GAME_VERSION = factory.GAME_VERSION
CHARACTER = factory.CHARACTER
ASCENSION = factory.ASCENSION

# ── reward (PREREG-0 "Reward") ───────────────────────────────────────────────

R_WIN = 1.0
R_BOSS = 0.05
R_FLOOR = 0.002

# ── caps (plan §3.2) ─────────────────────────────────────────────────────────

MAX_FIGHT_TURNS = C.MAX_FIGHT_TURNS  # 100
MAX_FIGHT_DECISIONS = C.MAX_FIGHT_DECISIONS  # 1,000
MAX_RUN_DECISIONS = C.MAX_RUN_DECISIONS  # 6,000
# All steps, forced ones included. Not a plan cap: a backstop for a forced-step loop the
# stored-decision caps cannot see. Invincible random play takes ~8,000 steps per run.
MAX_RUN_STEPS = 200_000

F9_P = SAMPLE_RATE

# ── decision types and action kinds (contracts.py §6, §8) ────────────────────

COMBAT = "combat"  # E2EDecision.kind of a combat row
OOC = "ooc"

# The run-level forms of combat actions (`RunManager.available_actions()` in a fight).
COMBAT_RUN_KINDS = frozenset(
    {
        "play_card",
        "end_turn",
        "drink_potion",
        "combat_resolve_choice",
        "combat_resolve_choice_take_first",
    }
)

# One key per segment for the F9 floor watermark (unique within the process).
_SEGMENT_SERIAL = itertools.count()


def dtype_name(dtype: int) -> str:
    """`contracts.DECISION_TYPES[dtype].name`, e.g. `COMBAT_TURN`, `NEOW`."""
    return C.DECISION_TYPES[int(dtype)].name


def decision_type(actions: list[dict[str, Any]], rm: Any, obs: dict[str, Any] | None = None) -> int:
    """The `contracts.DT` id of a run-level candidate list: `COMBAT_TURN` for a fight's
    run-level actions, else `classify_ooc` (raises `contracts.SchemaError`). Kept for
    callers of the earlier string form; `E2EDecision.dtype` is the same id."""
    if is_combat(actions):
        return C.DT["COMBAT_TURN"]
    return classify_ooc(actions, rm, obs or {})[0]


def is_combat(actions: list[dict[str, Any]]) -> bool:
    """Whether a run-level candidate list is a fight's (by kind, never by phase)."""
    return bool(actions) and all(a.get("action") in COMBAT_RUN_KINDS for a in actions)


def classify_ooc(
    actions: list[dict[str, Any]], rm: Any, obs: dict[str, Any]
) -> tuple[int, list[int]]:
    """`contracts.classify_ooc` with the run's context, called as the schema's live walk
    calls it (tests/e2e/test_contracts.py `_walk`). Raises `contracts.SchemaError`."""
    names = {a.get("action") for a in actions}
    event = rm.current_event_name() if "event_choice" in names else None
    grant_n = len(rm.card_grant_options()) if "grant_deck_card" in names else None
    bundles = bool(obs.get("card_reward_bundles")) and "pick_card_reward" in names
    return C.classify_ooc(actions, event_name=event, bundles=bundles, grant_n_options=grant_n)


def ooc_keys(actions: list[dict[str, Any]], kinds: list[int], obs: dict[str, Any]) -> list[str]:
    """`contracts.ooc_class_keys` over `contracts.ooc_payloads` (`obs` = the decision's
    `PpoRunEnv.observe()`)."""
    return C.ooc_class_keys(kinds, C.ooc_payloads(obs, actions, kinds))


# ── construction (plan §3.1: `ppo.e2e.factory` is the only construction site) ──


def make_run_manager(seed: int) -> Any:
    """The campaign `RunManager` for `seed`: `factory.new_run`, which builds and reads back
    Ironclad A10 on v0.111.0 in agent mode with the interactive potion swap."""
    return factory.new_run(int(seed))


def check_campaign_run(rm: Any) -> Any:
    """`factory.check_run`: raise unless `rm` reads back as a campaign run (build, mode,
    swap flag, character, ascension). Called on every run the env did not build."""
    return factory.check_run(rm)


# ── what ends a segment, and what ends the process ───────────────────────────


class NoActions(RuntimeError):
    """The engine offers no legal action in a run that is not over (`E2EEnv._stall`)."""


def is_rust_panic(ex: BaseException) -> bool:
    """Whether `ex` is a panic inside the PyO3 engine: `pyo3_runtime.PanicException`, a
    `BaseException` that no importable module exports, so it is matched by name (GH #335)."""
    return any(
        c.__module__ == "pyo3_runtime" and c.__name__ == "PanicException" for c in type(ex).__mro__
    )


def ends_segment(ex: BaseException) -> bool:
    """Whether `ex`, raised while the env drives a segment, ends that segment as an error
    terminal (`E2EEnv._error`, counted `error/<type>`) rather than propagating.

    G-schema: an engine error or an assert, the tokenizer's and the schema's included, is
    an errored run, not a dead worker; so is a Rust panic, although it is a
    `BaseException`. A build mismatch is not: a `sts2_client.GameVersionMismatch` (an
    `AssertionError`) from the factory's fight-start or run check means the process or the
    engine is on the wrong build, so every later segment would fail the same way; it
    propagates. So does every other `BaseException` (`KeyboardInterrupt`, `SystemExit`,
    `GeneratorExit`). The learner's `GuardStop` and `NanStop` are raised in the learner
    process, which runs no env.
    """
    if isinstance(ex, sts2_client.GameVersionMismatch):
        return False
    return isinstance(ex, Exception) or is_rust_panic(ex)


# ── dedupe (plan §3.2) ───────────────────────────────────────────────────────

# Per-copy card variables that change the card face: the per-copy damage or block a played
# copy carries (Rampage +5 per play, Claw, Genetic Algorithm, Ritual Dagger, momentum), a
# combat-long Ethereal, and a spent one-shot enchantment (`enchantment_used`: Vigorous adds
# its amount until the card's first play, C# `Vigorous.AfterCardPlayed`; an Anger clone,
# made in `OnPlay` before that, keeps it, so a hand can hold a spent and a fresh copy that
# preview different damage, #1097). Cost variables are already in the effective `cost`.
FACE_VARS = (
    "rampage_damage",
    "claw_buff",
    "genetic_algorithm_block",
    "ritual_dagger_damage",
    "momentum_damage",
    "ethereal_this_combat",
    "enchantment_used",
)


# Two shortcuts that change no stored row, class or trajectory (tests/e2e/test_env.py plays
# whole segments both ways): a forced combat step inside a round that the fight's
# `IntentLog` has already logged skips the honest snapshot, and a choose whose selected
# content no other choose shares skips its outcome snapshot. Together they cut the env's
# time per stored row by 27% on HP-pinned random play (450 to 329 us). False restores the
# plain path.
FAST_PATHS = True


def _face_vars(raw_card: dict[str, Any]) -> dict[str, Any]:
    cv = raw_card.get("combat_vars") or {}
    return {k: cv[k] for k in FACE_VARS if cv.get(k)}


def _shared_selections(choice: Any, acts: Any, piles: dict[str, Any]) -> set[int]:
    """The choose actions whose selected content (as `contracts.combat_class_keys` keys
    it) another choose action shares. Only these can merge, so only their outcome counts."""
    sel = {
        i: tuple(
            sorted(C.choice_option_key(choice, int(x), piles) for x in a.get("selection") or [])
        )
        for i, a in enumerate(acts)
        if a.get("action") == "choose"
    }
    n = collections.Counter(sel.values())
    return {i for i, k in sel.items() if n[k] > 1}


def combat_keys(
    bv: Any, d_bv: dict[str, Any], acts: Any, legacy_acts: list[dict[str, Any]] | None = None
) -> list[str]:
    """One merge key per action: `contracts.combat_class_keys`, with each play key refined
    by the card's per-copy face variables (`FACE_VARS`; the T6 deviation in the module
    docstring). `acts` must be in `bv`'s index space (`snapshot.obs_actions`).

    A choose keys on its outcome and its selected content, so its outcome only matters
    when another choose selects the same content. With `FAST_PATHS` the env computes it
    only then: the classes are the schema's, and the key of a choose that is a class of its
    own carries an empty outcome.

    The earlier form `combat_keys(bv, d_bv, snap, acts)` still works (the snapshot is
    unused: the schema's keys read the belief view's raw hand).
    """
    from ppo.combat_ctx.snapshot import choose_outcome_key

    if legacy_acts is not None:
        acts = legacy_acts

    deck = d_bv["deck"]
    hand = deck.get("hand", [])
    pots = d_bv["player"].get("potions", [])
    choice = bv.pending_choice()
    shared = _shared_selections(choice, acts, deck) if FAST_PATHS and choice else None

    def outcome(i: int) -> str:
        if shared is not None and i not in shared:
            return ""  # no other choose selects this content: a class of its own
        c = bv.clone()
        c.apply(acts[i])
        return choose_outcome_key(c)

    keys = C.combat_class_keys(acts, hand, pots, choice=choice, outcome=outcome, piles=deck)
    out = []
    for k, a in zip(keys, acts, strict=True):
        if a.get("action") == "play":
            i = int(a["card_index"])
            fv = _face_vars(hand[i]) if i < len(hand) else {}
            if fv:
                k = json.dumps([k, fv], sort_keys=True, default=str)
        out.append(k)
    return out


def classes_from_keys(keys: list[Any]) -> tuple[list[int], list[list[int]]]:
    """Dense class ids in first-occurrence order, and each class's members in action order."""
    ids: dict[Any, int] = {}
    cls = [ids.setdefault(k, len(ids)) for k in keys]
    members: list[list[int]] = [[] for _ in ids]
    for i, c in enumerate(cls):
        members[c].append(i)
    return cls, members


def combat_view(cs: Any, room: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """What a combat row is built from: `(bv, sts2_rs.state_to_dict(bv), snapshot)`.

    `bv` is `cs.belief_view(0)` and the snapshot is `snapshot_from_parts` over it (the
    honest hand with its damage and block previews, the pending choice), before the
    fight's `IntentLog` adds `ehist`.
    """
    bv, d_bv = _belief(cs)
    return bv, d_bv, _snapshot(bv, d_bv, room)


def _belief(cs: Any) -> tuple[Any, dict[str, Any]]:
    """`(cs.belief_view(0), state_to_dict(bv))`. The view leaves `cs` as it was."""
    import sts2_rs

    bv = cs.belief_view(0)
    return bv, sts2_rs.state_to_dict(bv)


def _snapshot(bv: Any, d_bv: dict[str, Any], room: str) -> dict[str, Any]:
    from ppo.combat_ctx.snapshot import snapshot_from_parts

    return snapshot_from_parts(
        d_bv, bv.card_damage_preview(), bv.card_block_preview(), bv.pending_choice(), room
    )


@dataclass(frozen=True)
class CombatClasses:
    """One combat state classified and deduped as the env does (`merged_classes`)."""

    dtype: int  # contracts.DT: COMBAT_TURN or COMBAT_CHOICE
    kinds: list[int]  # contracts action kind per member
    obs_actions: list[dict[str, Any]]  # the members in the belief view's index space
    classes: list[int]  # merged class per member
    members: list[list[int]]  # member indices per class, in order
    keys: list[str] = field(default_factory=list)  # dedupe key per member (`combat_keys`)


def merged_classes(
    cs: Any,
    legal: list[dict[str, Any]] | None = None,
    *,
    room: str = "monster",
    view: tuple[Any, dict[str, Any], dict[str, Any]] | None = None,
) -> CombatClasses:
    """Classify and dedupe one combat state exactly as the env does.

    `legal` defaults to `cs.legal_actions()`; `obs_actions[i]` is `legal[i]` in the belief
    view's index space. `view` reuses a `combat_view(cs, room)`. Raises
    `contracts.SchemaError` on an action kind the schema does not know.
    """
    from ppo.combat_ctx.snapshot import obs_actions

    legal = cs.legal_actions() if legal is None else legal
    bv, d_bv, _snap = view if view is not None else combat_view(cs, room)
    dtype, kinds = C.classify_combat(legal, cs.pending_choice())
    acts = obs_actions(legal, cs.pending_choice(), bv.pending_choice())
    keys = combat_keys(bv, d_bv, acts)
    cls, members = classes_from_keys(keys)
    return CombatClasses(dtype, kinds, acts, cls, members, keys)


def combat_selection(choice: dict[str, Any] | None, kinds: list[int]) -> C.Selection:
    """The open selection of a combat choice (SCHEMA.md §2.4), from `bv.pending_choice()`
    and the members' kinds: the engine offers one CHOOSE per still-unpicked option, so the
    options without one are the picks made."""
    if choice is None:
        return C.NO_SELECTION
    opts = choice.get("cards") or choice.get("options") or []
    return C.Selection(
        min=int(choice.get("min_select") or 0),
        max=int(choice.get("max_select") or 0),
        picked=len(opts) - kinds.count(C.AK["CHOOSE"]),
        can_skip=C.AK["CHOOSE_DONE"] in kinds,
    )


def check_caps(
    n_members: int, n_classes: int, what: str, counts: collections.Counter | None = None
) -> None:
    """The schema's per-decision caps (SCHEMA.md §7): 255 members, 64 merged options.

    SCHEMA.md asserts on a breach, so the segment ends as an error. `counts` gets
    `assert/members` or `assert/classes` first, so that the pre-flight can name the cause.
    The member cap is reachable: an HP-pinned random run (seed 1300100278) reached round
    88 of a fight with 256 Anger copies in the discard pile, and a put-back choice then
    offered 266 cards in 12 classes.
    """
    over = [
        name
        for name, n, cap in (
            ("members", n_members, C.MAX_MEMBERS),
            ("classes", n_classes, C.MAX_OPTIONS),
        )
        if n > cap
    ]
    if over:
        if counts is not None:
            for name in over:
                counts[f"assert/{name}"] += 1
        raise C.SchemaError(
            f"{what}: {n_members} members / {n_classes} classes exceed "
            f"{C.MAX_MEMBERS} / {C.MAX_OPTIONS}"
        )


# ── the run stepper ──────────────────────────────────────────────────────────


class _RunEnv(PpoRunEnv):
    """`PpoRunEnv(combat="yield")` over a factory-built run, with one change of behaviour.

    `_advance` is the parent's (ppo/env.py:568-690), except that:

    * a fight is recognised by its action kinds (`is_combat`), not by `phase()`, so a
      deck operation or potion swap that preempts a fight is an OOC decision;
    * a full-belt potion swap is always a decision: the campaign student decides it
      (D4), so the rule-side branch (#479, env.py:220-232) is refused at construction;
    * every Move goes through `_take`, which hands the pre-move clone to `move_hook`
      (practice capture, plan §3.3), forced moves included, and every action then calls
      `step_hook` (F9);
    * no per-step resolution log (a run has ~10^4 steps).
    """

    move_hook: Callable[[Any, dict[str, Any]], None] | None
    step_hook: Callable[[], None] | None
    grant_picked: int  # card-grant picks applied since the grant opened

    @classmethod
    def wrap(cls, rm: Any, seed: int, *, max_steps: int) -> _RunEnv:
        import sts2_rs

        if sts2_rs.potion_swap_is_rule_side():
            raise RuntimeError(
                "STS2_RUNAI_POTION_OC is set, so PpoRunEnv would resolve full-belt potion "
                "swaps rule-side; the campaign student decides them (plan D4). Unset it."
            )
        env = object.__new__(cls)
        env.seed = int(seed)
        env.ascension = int(rm.ascension_level)
        env.character = factory.CHARACTER
        env.sims = 0
        env.boss_sims = 0
        env.max_steps = int(max_steps)
        env.expert = False
        env.combat = "yield"
        env.surrogate_path = None
        env.combat_ckpt = None
        env.act1 = str(rm.act1)
        env.game_version = str(rm.game_version)
        env.curse_inject_p = 0.0
        env.injected_curse = None
        env.resolution_log = []
        env.rm = rm
        env.agent = None
        env._surrogate = None
        env.steps = 0
        env.decisions = 0
        env._pending_actions = None
        env.move_hook = None
        env.step_hook = None
        env.forced = collections.Counter()
        env.grant_picked = 0
        return env

    def _take(self, action: dict[str, Any]) -> Any:
        if action.get("action") == "grant_deck_card":
            # A pick, not Sea Glass's confirm (`idx == len(options)`): `Selection.picked`.
            if int(action.get("idx") or 0) < len(self.rm.card_grant_options()):
                self.grant_picked += 1
        else:
            self.grant_picked = 0  # any other action ends the grant
        hook = self.move_hook
        if hook is not None and action.get("action") == "move":
            before = self.rm.clone()
            res = self.rm.take_action(action)
            hook(before, action)
        else:
            res = self.rm.take_action(action)
        if self.step_hook is not None:
            self.step_hook()
        return res

    def combat_step(self) -> None:
        """Count one combat action the caller applied (the parent's, minus its log)."""
        self.steps += 1

    def _advance(self, mode: str) -> RunDecision | CombatTurn | Terminal:
        rm = self.rm
        while True:
            if rm.is_over():
                return self._terminal("lost")
            if self.steps >= self.max_steps:
                return self._terminal("max_steps")
            phase = rm.phase()
            if phase == "event":
                rm.materialize_event()  # #302: bind before the options are counted
            actions = rm.available_actions()
            if not actions:
                return self._terminal("no_actions")
            actions = self._mask_rejectable(phase, actions)
            if is_combat(actions):
                return CombatTurn(step_index=self.steps)
            if rm.pending_claws_select:
                self.forced["CARD_TRANSFORM"] += 1
                self._take(_claws_auto_pick(actions))
                self.steps += 1
                continue
            assert_encodable(actions, phase=phase)
            surface = classify_surface(phase, actions)
            if len(actions) == 1:
                self.forced[_forced_name(actions, rm)] += 1
                self._take(actions[0])
                self.steps += 1
                continue
            obs = self.observe(phase, surface)
            metrics = self.metrics_snapshot()
            self.decisions += 1
            self._pending_actions = actions
            return RunDecision(
                surface=surface or "unknown",
                phase=phase,
                actions=actions,
                obs=obs,
                metrics=metrics,
                step_index=self.steps,
            )

    def step(self, idx: int) -> RunDecision | CombatTurn | Terminal:
        if self._pending_actions is None:
            raise RuntimeError("step() without a pending decision")
        actions = self._pending_actions
        if not 0 <= idx < len(actions):
            raise IndexError(f"candidate idx {idx} out of range 0..{len(actions) - 1}")
        self._pending_actions = None
        result = self._take(actions[idx])
        if result in ("shop_purchase_failed", "shop_remove_card_failed", "invalid_for_phase"):
            raise RuntimeError(f"engine rejected action {actions[idx]!r}: {result}")
        self.steps += 1
        return self._advance("net")


def _claws_auto_pick(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """The forced step that resolves Claws' optional "up to 6 into Maul" select.

    GH #1353: the first offered card until the cap, then the confirm, which is
    always last. That is the first six candidates in deck order, the pick the
    engine made before the select surfaced. The schema has no kind for the
    confirm, so the select stays out of the policy until a schema revision adds
    one. Mirrors `claws_auto_pick` in `sts2-e2e/src/env.rs`.
    """
    return actions[0]


def _forced_name(actions: list[dict[str, Any]], rm: Any) -> str:
    """The decision type of a single-option step, for the `forced/<type>` counters."""
    try:
        return dtype_name(classify_ooc(actions, rm, {})[0])
    except C.SchemaError:
        return f"unknown_kind:{actions[0].get('action')}"


# ── yielded objects ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Segment:
    """What an `E2EEnv` walks: a whole run, or a practice segment (plan §3.3).

    `end` is `terminal` (play to the run's end: full runs and double-boss segments) or
    `first_ooc` (stop at the first OOC decision, a value-only bootstrap row).
    """

    kind: str = "run"
    cell: str | None = None  # a `contracts.CELLS` name for practice
    end: str = "terminal"
    parent_seed: int | None = None
    parent_idx: tuple[int, ...] = ()
    capture_steps: int | None = None
    move: tuple[tuple[str, Any], ...] | None = None
    reseed: int | None = None
    uid: int | None = None  # the row batch's `run_uid`; None: the run seed

    @property
    def cell_id(self) -> int:
        """The row batch's `cell`: the index into `contracts.CELLS`, 255 outside practice."""
        return C.CELLS.index(self.cell) if self.cell is not None else C.CELL_NONE

    def header(self) -> dict[str, Any]:
        h: dict[str, Any] = {"kind": self.kind, "cell": self.cell, "end": self.end}
        if self.kind != "run":
            h.update(
                parent_seed=self.parent_seed,
                parent_idx=list(self.parent_idx),
                capture_steps=self.capture_steps,
                move=dict(self.move or ()),
                reseed=self.reseed,
                uid=self.uid,
            )
        return h


RUN = Segment()


@dataclass(eq=False)
class E2EDecision:
    """One stored decision: SCHEMA.md §1 "one row per decision".

    The actor answers with a class index into `members` (the row batch's `action`); the
    env applies the class's representative, its first member. Members of a class
    tokenize byte-identically (the tokenizer asserts it). `classes` is the row batch's
    `mem_class` and `cls_kind` its `cls_kind`.

    `decision` is the same decision as the schema's `contracts.EnvDecision` (SCHEMA.md
    §8.1), the tokenizer's input. The tokenizer reads `policy()` (a
    `contracts.PolicySource`), the value path also `priv()` (a `contracts.PrivSource`).
    Live references (`rm`, `cs`, `bv`) are valid only until the next `step`; consumers
    read them and never mutate them.
    """

    row: int  # index of this stored row in the segment: the row batch's `t`
    kind: str  # COMBAT | OOC
    dtype: int  # contracts.DT id
    kinds: list[int]  # contracts action kind per member
    actions: list[dict[str, Any]]  # combat: cs.legal_actions() (true state); OOC: run actions
    obs_actions: list[dict[str, Any]]  # combat: in bv's option order (obs_actions); OOC: actions
    classes: list[int]  # merged-class id per member
    members: list[list[int]]  # member indices per class, in member order
    keys: list[str]  # dedupe key per member (`combat_keys`, `ooc_keys`)
    reward: float  # reward just assigned to the previous stored row
    unit: tuple[str, int]  # ("fight", k) or ("act", a): the teacher-acting unit
    unit_start: bool  # first stored row of `unit`
    act: int
    floor: int
    rm: Any
    seed: int
    segment: Segment
    bootstrap: bool = False  # value-only row: no action is taken, the segment ends
    sel: C.Selection = C.NO_SELECTION  # the open selection (COMBAT_CHOICE, CARD_GRANT)
    event_name: str | None = None  # rm.current_event_name() on EVENT and NEOW
    # combat only
    cs: Any = None
    bv: Any = None
    d_bv: dict[str, Any] | None = None  # sts2_rs.state_to_dict(bv)
    snap: dict[str, Any] | None = None  # honest snapshot with this fight's `ehist`
    room: str | None = None  # monster / elite / boss
    fight: int | None = None  # fight sequence number in the segment
    turn: int | None = None  # the fight's round counter
    fight_row: int | None = None  # stored rows of this fight before this one
    # OOC only
    run_obs: dict[str, Any] | None = None  # ppo.env.observe(...): the run teacher's input
    metrics: dict[str, Any] | None = None
    surface: str | None = None  # ppo.env.classify_surface (coarse)
    fine_surface: str | None = None  # ppo.encoder.fine_surface

    @property
    def n_classes(self) -> int:
        return len(self.members)

    @property
    def is_combat(self) -> bool:
        return self.kind == COMBAT

    @property
    def dtype_name(self) -> str:
        return dtype_name(self.dtype)

    @property
    def cls_kind(self) -> list[int]:
        """The action kind of each class (its representative's)."""
        return [self.kinds[m[0]] for m in self.members]

    @property
    def run_uid(self) -> int:
        return self.segment.uid if self.segment.uid is not None else self.seed

    @property
    def cell(self) -> int:
        return self.segment.cell_id

    def policy(self) -> PolicySource:
        """What the policy tokenizer may read (SCHEMA.md §8)."""
        return PolicySource(self.bv, self.obs_actions, self.rm)

    def priv(self) -> PrivSource:
        """What only the value path may read (SCHEMA.md §8)."""
        return PrivSource(self.cs, self.rm)

    def meta(self) -> RowMeta:
        """The fields `segment_rows` reads, without the live references."""
        return RowMeta(
            run_uid=self.run_uid,
            t=self.row,
            is_combat=self.is_combat,
            dtype=self.dtype,
            bootstrap=self.bootstrap,
            n_classes=self.n_classes,
            n_mem=len(self.actions),
            cell=self.cell,
            classes=tuple(self.classes),
            cls_kind=tuple(self.cls_kind),
        )

    @functools.cached_property
    def decision(self) -> C.EnvDecision:
        """This decision as the schema's `contracts.EnvDecision` (SCHEMA.md §8.1).

        Its `actions` are the members as the policy sees them: the belief view's index
        space in combat (`obs_actions`), the run actions out of combat. So
        `member_action(c)` is not a live action in combat: act with `E2EEnv.step(c)`.
        Built on first use; valid, like the live references, until the next `step`.
        """
        cls, rep = C.merge_classes(self.keys)
        combat = self.is_combat
        return C.EnvDecision(
            run_uid=self.run_uid,
            t=self.row,
            dtype=self.dtype,
            actions=self.obs_actions,
            kinds=np.asarray(self.kinds, np.uint8),
            keys=list(self.keys),
            member_class=cls,
            rep=rep,
            policy=self.policy(),
            priv=self.priv(),
            sel=self.sel,
            event_name=self.event_name,
            obs=None if combat else self.run_obs,
            metrics=None if combat else self.metrics,
            cell=self.cell,
            fight_turn=int(self.turn or 0) if combat else 0,
            fight_decision=int(self.fight_row or 0) if combat else 0,
        )


@dataclass(frozen=True)
class RowMeta:
    """What `segment_rows` reads from one stored decision. The env keeps one per row
    (`E2EEnv.row_meta`), so the actor can drop each decision once it has tokenized it: a
    kept `E2EDecision` holds about 75 KB of Python objects and two combat states (on an
    HP-pinned random run, 854 kept decisions took 62 MB of heap and 177 MB of RSS)."""

    run_uid: int
    t: int
    is_combat: bool
    dtype: int
    bootstrap: bool
    n_classes: int
    n_mem: int
    cell: int
    classes: tuple[int, ...]
    cls_kind: tuple[int, ...]


# The run-level reads the policy may make: `contracts.POLICY_CALLS`' `rm.*` names, and
# the ones whose raw value carries hidden information (`contracts.POLICY_FILTERS`).
# A fill event adds its own (`contracts.FILL_POLICY_CALLS`: "chars" reads `rm.character`).
_POLICY_RM = frozenset(
    c.split(".", 1)[1]
    for c in (*C.POLICY_CALLS, *(c for cs in C.FILL_POLICY_CALLS.values() for c in cs))
    if c.startswith("rm.")
)
_POLICY_RM_FILTERS = {
    c.split(".", 1)[1]: f for c, f in C.POLICY_FILTERS.items() if c.startswith("rm.")
}


class RunFacade:
    """A read-only view of the run: only the `rm.*` reads in `contracts.POLICY_CALLS`
    (`materialize_event` included: it binds an event already bound). A read in
    `contracts.POLICY_FILTERS` returns its filter's output, never the raw value:
    `event_option_infos()` gives `honest_event_options(...)` (SCHEMA.md §8). A read the
    wheel lacks raises `AttributeError`, so callers can use `hasattr`."""

    __slots__ = ("_rm",)

    def __init__(self, rm: Any) -> None:
        object.__setattr__(self, "_rm", rm)

    def __getattr__(self, name: str) -> Any:
        if name not in _POLICY_RM:
            raise AttributeError(f"{name!r} is not a policy read (contracts.POLICY_CALLS)")
        raw = getattr(object.__getattribute__(self, "_rm"), name)
        flt = _POLICY_RM_FILTERS.get(name)
        if flt is None:
            return raw
        if callable(raw):
            return lambda *args, **kwargs: flt(raw(*args, **kwargs))
        return flt(raw)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("the run facade is read-only")


class PolicySource:
    """`contracts.PolicySource` for one decision or combat step.

    `PolicySource(bv, legal, rm)`. The earlier one-argument form `PolicySource(d)` still
    works for any object with `bv`, `obs_actions` and `rm` (an `E2EDecision`, or the
    tokenizer's `RecordedDecision`)."""

    __slots__ = ("_bv", "_legal", "_rm")

    def __init__(self, bv: Any, legal: list[dict[str, Any]] | None = None, rm: Any = None) -> None:
        if legal is None and rm is None:  # the earlier form: one decision
            bv, legal, rm = bv.bv, bv.obs_actions, bv.rm
        self._bv = bv
        self._legal = legal
        self._rm = rm

    def belief(self) -> Any:
        """`cs.belief_view(0)` in combat, else None."""
        return self._bv

    def legal(self) -> list[dict[str, Any]]:
        """The members, in the belief view's index space in combat."""
        return self._legal

    def run(self) -> RunFacade:
        return RunFacade(self._rm)


class PrivSource:
    """`contracts.PrivSource` for one decision: privileged reads for the value path.

    `PrivSource(cs, rm)`. The earlier one-argument form `PrivSource(d)` still works for
    any object with `cs` and `rm`."""

    __slots__ = ("_cs", "_rm")

    def __init__(self, cs: Any, rm: Any = None) -> None:
        if rm is None:  # the earlier form: one decision
            cs, rm = cs.cs, cs.rm
        self._cs = cs
        self._rm = rm

    def true_state(self) -> Any:
        """The true combat state (`cs`), or None out of combat."""
        return self._cs

    def act_rooms(self) -> Mapping[str, Any]:
        """This act's `generated_rooms()` record (`generated_rooms()[act_index - 1]`, keys
        `normal`, `elite`, `events_shuffled`, `boss`, `second_boss`), or `{}` past the
        last act. The record itself, as `tokens.RunPriv` returns it: PRIV_RUN counts its
        queues and pools its encounters."""
        rooms = self._rm.generated_rooms()
        a = int(self._rm.act_index) - 1
        return rooms[a] if 0 <= a < len(rooms) else {}

    def future(self) -> list[Mapping[str, Any]]:
        """The later acts' `generated_rooms()` records (`generated_rooms()[act_index:]`),
        whose bosses PRIV_RUN reads.

        No future maps: schema revision 2 lists `rm.act_map_for` as a PRIV call that no
        PRIV token reads at launch (SCHEMA.md §2.10, open question 17)."""
        rm = self._rm
        return list(rm.generated_rooms()[int(rm.act_index) :])


@dataclass(eq=False)
class E2ETerminal:
    """The end of a segment. `reward` is the last stored row's final increment."""

    outcome: str  # won | lost | capped | bootstrap | error
    won: bool
    cap: str | None
    reward: float
    total_reward: float
    rewards: list[float]
    unassigned_reward: float
    rows: int
    steps: int
    floor: int
    act: int
    seed: int
    segment: Segment
    bosses: int
    reached_boss2: bool
    fingerprint: str
    log_idx: list[int]
    counts: collections.Counter
    error: str | None = None


# ── fights ───────────────────────────────────────────────────────────────────

ROOMS = ("monster", "elite", "boss")


def fight_room(rm: Any) -> str:
    room = (rm.current_room() or {}).get("room_type")
    return room if room in ROOMS else "monster"


@dataclass
class _Fight:
    seq: int
    room: str
    where: tuple[Any, ...]
    act: int
    floor: int
    ilog: Any
    round: int = 0
    rows: int = 0
    steps: int = 0


def run_fingerprint(rm: Any) -> str:
    """A digest of the run's observable and RNG state (replay checks, DEVBOSS storage)."""
    d: dict[str, Any] = {
        "phase": rm.phase(),
        "over": bool(rm.is_over()),
        "won": bool(rm.player_won),
        "act": int(rm.act_index),
        "floor": int(rm.total_floor()),
        "hp": int(rm.current_hp),
        "max_hp": int(rm.max_hp),
        "gold": int(rm.gold),
        "deck": [list(x) for x in rm.master_deck_detailed()],
        "ench": [list(x) if x else None for x in rm.master_deck_enchantments()],
        "relics": [str(x) for x in rm.relics()],
        "potions": [str(x) for x in rm.potions()],
        "rng": rm.rng_counters(),
        "coord": rm.map_coord(),
    }
    cs = rm.combat_state() if not rm.is_over() else None
    if cs is not None:
        # The full ordered combat state (piles in order). Not `rng_fingerprint`, which
        # raises on a raw state with a hidden draw pile of two or more cards.
        d["combat"] = int(cs.ordered_hash())
    blob = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


# ── the env ──────────────────────────────────────────────────────────────────


@dataclass
class EnvConfig:
    max_fight_turns: int = MAX_FIGHT_TURNS
    max_fight_decisions: int = MAX_FIGHT_DECISIONS
    max_run_decisions: int = MAX_RUN_DECISIONS
    max_run_steps: int = MAX_RUN_STEPS
    f9_p: float = F9_P
    # One checker per worker, shared by its envs, so its counters cover the worker
    # (`InvariantChecker.stats()`). None: each env makes its own at `f9_p`.
    f9: InvariantChecker | None = None
    # `invariants.KNOWN_ISSUES` keys the env's own checker counts as tolerated instead of
    # as violations (`f9/tolerated/known_<key>`). Empty, the default, is strict.
    f9_tolerate: tuple[str, ...] = ()
    # Feed practice: called with a `Capture` before every Move into an elite or boss
    # node of a full run (plan §3.3). None turns capture off (eval runs, practice).
    on_capture: Callable[[Capture], None] | None = None
    # A `contracts.Tokenizer` (WP3b) the env drives: `reset_fight()` at every fight
    # start, `observe(policy)` at every combat step before the action (SCHEMA.md §8.2).
    tokenizer: Any = None
    # Exposed for tests only: skip the factory's version guard.
    allow_any_run: bool = False
    # `new_env` builds an `rsenv.RsE2EEnv` (the Rust training env) instead of an
    # `E2EEnv`, its rows tokenized with this `sts2_rs.E2eRareMap` (None: no RareMap).
    rust: bool = False
    rust_rare: Any = None


@dataclass
class Capture:
    """A practice snapshot: the run just before the Move into an elite or boss room."""

    cell: str  # a contracts.CELLS name
    snap: Any  # rm.clone() before the Move
    move: dict[str, Any]
    seed: int
    steps: int  # the run env's step count at the Move (replay key)
    rows: int  # stored rows before the Move
    idx: tuple[int, ...]  # the action log so far (replay prefix)


class E2EEnv:
    """One segment (a run, or a practice segment) as a stream of stored decisions."""

    def __init__(
        self,
        seed: int,
        *,
        rm: Any = None,
        cfg: EnvConfig | None = None,
        segment: Segment = RUN,
    ) -> None:
        self.cfg = cfg or EnvConfig()
        self.seed = int(seed)
        self.segment = segment
        if rm is None:
            rm = make_run_manager(self.seed)
        if not self.cfg.allow_any_run:
            check_campaign_run(rm)
        self.run = _RunEnv.wrap(rm, self.seed, max_steps=self.cfg.max_run_steps)
        self.rm = rm
        if self.cfg.on_capture is not None and segment.kind == "run":
            self.run.move_hook = self._on_move
        # rows and reward
        self.rows = 0
        self.rewards: list[float] = []
        self.total_reward = 0.0
        self.unassigned_reward = 0.0
        self._pending_r = 0.0
        self.max_floor = int(rm.total_floor())
        self.bosses = 0
        self.reached_boss2 = False
        # fights and units
        self.fight: _Fight | None = None
        self.fights = 0
        self._units: set[tuple[str, int]] = set()
        # bookkeeping
        self.row_meta: list[RowMeta] = []  # one per stored row: `segment_rows` input
        self.log_idx: list[int] = []
        self.counts: collections.Counter = collections.Counter()
        self._pending: E2EDecision | None = None
        self._done = False
        self._act3_bosses = 0
        self.f9 = self.cfg.f9
        if self.f9 is None:
            self.f9 = InvariantChecker(
                sample_rate=self.cfg.f9_p,
                seed=(self.seed << 32) ^ (int(segment.reseed or 0) & 0xFFFFFFFF),
                tolerate_known=self.cfg.f9_tolerate,
            )
        self._f9_key = ("e2e", next(_SEGMENT_SERIAL))  # floor watermark key: one per segment
        self.f9_violations: list[str] = []
        self.run.step_hook = self._f9_step

    # ── public ──────────────────────────────────────────────────────────────

    def reset(self) -> E2EDecision | E2ETerminal:
        return self._drive(self.run.reset())

    def step(self, choice: int) -> E2EDecision | E2ETerminal:
        """Apply merged class `choice` of the pending decision (its first member)."""
        d = self._pending
        if d is None:
            raise RuntimeError("step() without a pending decision")
        return self.step_action(d.members[int(choice)][0])

    def step_action(self, ai: int) -> E2EDecision | E2ETerminal:
        """Apply action `ai` (an index into the pending decision's `actions`)."""
        from ppo.combat_bench.oracle import combat_to_rm

        d = self._pending
        if d is None:
            raise RuntimeError("step_action() without a pending decision")
        if d.bootstrap:
            raise RuntimeError("a bootstrap row takes no action: the segment has ended")
        ai = int(ai)
        if not 0 <= ai < len(d.actions):
            raise IndexError(f"action {ai} out of range 0..{len(d.actions) - 1}")
        self._pending = None
        self.log_idx.append(ai)
        try:
            if d.kind == COMBAT:
                self._combat_apply(combat_to_rm(d.actions[ai]))
                out: Any = CombatTurn(step_index=self.run.steps)
            else:
                out = self.run.step(ai)
        except BaseException as ex:  # a Rust panic is a BaseException: `ends_segment`
            if not ends_segment(ex):
                raise
            return self._error(ex)
        return self._drive(out)

    @property
    def done(self) -> bool:
        return self._done

    def segment_rows(self) -> dict[str, np.ndarray]:
        """The env's row-batch fields of the finished segment (`segment_rows`), from
        `row_meta`: the actor need not keep the decisions."""
        if not self._done:
            raise RuntimeError("segment_rows() before the segment ended")
        return segment_rows(self.row_meta, self.terminal)

    # ── the drive loop ──────────────────────────────────────────────────────

    def _drive(self, out: Any) -> E2EDecision | E2ETerminal:
        try:
            while True:
                if isinstance(out, Terminal):
                    return self._terminal(out.outcome, won=bool(out.won))
                if isinstance(out, RunDecision):
                    res = self._ooc(out)
                    if isinstance(res, (E2EDecision, E2ETerminal)):
                        return res
                    out = res  # one merged class: the run stepper's next output
                    continue
                res = self._combat()
                if res is not None:
                    return res
                out = self.run.resume()
        except BaseException as ex:  # a Rust panic is a BaseException: `ends_segment`
            if not ends_segment(ex):
                raise
            return self._error(ex)

    def _error(self, ex: BaseException) -> E2ETerminal:
        """End the segment as an error with a counted reason (G-schema: an assert, the
        tokenizer's included, is an errored run, not a dead worker): `error/<type>`, for
        example `error/SchemaError` or `error/PanicException`."""
        import traceback

        self.counts["error"] += 1
        self.counts[f"error/{type(ex).__name__}"] += 1
        self.fight = None
        return self._terminal("error", won=False, error=traceback.format_exc(limit=6))

    def _combat(self) -> E2EDecision | E2ETerminal | None:
        """Play forced combat steps until a stored decision (returned), or until the run
        leaves the fight (None: resume the run stepper)."""
        rm = self.rm
        cfg = self.cfg
        while True:
            if rm.is_over():
                if self.fight is not None:
                    self._end_fight(bool(rm.player_won))
                return None
            run_acts = rm.available_actions()
            if not is_combat(run_acts):
                if self.fight is not None and rm.phase() != "combat":
                    self._end_fight(True)
                return None  # an OOC decision preempts, or the fight is over
            if self.run.steps >= cfg.max_run_steps:
                return self._cap("steps")
            cs = rm.combat_state()
            legal = cs.legal_actions()
            if not legal:
                raise RuntimeError("the run offers combat actions but the fight has none")
            if self.fight is None:
                self._start_fight(cs)
            f = self.fight
            assert f is not None
            where = (int(rm.act_index), rm.map_coord())
            if where != f.where:
                self._end_fight(True)
                self._start_fight(cs)
                f = self.fight
                assert f is not None
            bv, d_bv = _belief(cs)
            rnd = int(d_bv.get("round") or 0)  # the snapshot's `round`
            if rnd < f.round:
                # A new fight in the same combat phase (runslot.py:287-292).
                self.counts["round_reset"] += 1
                self._end_fight(True)
                self._start_fight(cs)
                f = self.fight
                assert f is not None
            f.round = rnd
            if rnd > cfg.max_fight_turns:
                return self._cap("turns")
            if cfg.tokenizer is not None:
                from ppo.combat_ctx.snapshot import obs_actions

                acts = obs_actions(legal, cs.pending_choice(), bv.pending_choice())
                cfg.tokenizer.observe(PolicySource(bv, acts, rm))
            if len(legal) == 1 and FAST_PATHS and f.ilog.last_round == rnd:
                # Forced, and the log has this round: `observe` would only attach `ehist`
                # to a snapshot that no row keeps.
                self.counts["forced/combat"] += 1
                self._combat_apply(_combat_to_rm(legal[0]))
                continue
            snap = f.ilog.observe(_snapshot(bv, d_bv, f.room))
            if len(legal) == 1:
                self.counts["forced/combat"] += 1
                self._combat_apply(_combat_to_rm(legal[0]))
                continue
            try:
                cc = merged_classes(cs, legal, view=(bv, d_bv, snap))
            except C.SchemaError:
                self.counts["unknown"] += 1
                raise
            members = cc.members
            if len(members) == 1:
                self.counts["forced/combat_merged"] += 1
                self._combat_apply(_combat_to_rm(legal[members[0][0]]))
                continue
            check_caps(len(legal), len(members), dtype_name(cc.dtype), self.counts)
            if f.rows >= cfg.max_fight_decisions:
                return self._cap("fight_decisions")
            if self.rows >= cfg.max_run_decisions:
                return self._cap("run_decisions")
            if len(legal) > len(members):
                self.counts["merged_rows"] += 1
                self.counts["merged_actions"] += len(legal) - len(members)
            sel = C.NO_SELECTION
            if cc.dtype in C.SELECTION_DTYPES:
                sel = combat_selection(bv.pending_choice(), cc.kinds)
            d = self._emit(
                kind=COMBAT,
                dtype=cc.dtype,
                kinds=cc.kinds,
                actions=legal,
                obs_actions=cc.obs_actions,
                classes=cc.classes,
                members=members,
                keys=cc.keys,
                sel=sel,
                unit=("fight", f.seq),
                cs=cs,
                bv=bv,
                d_bv=d_bv,
                snap=snap,
                room=f.room,
                fight=f.seq,
                turn=rnd,
                fight_row=f.rows,
            )
            f.rows += 1
            return d

    def _combat_apply(self, run_action: dict[str, Any]) -> None:
        rm = self.rm
        rm.take_action(run_action)
        self.run.combat_step()
        if self.fight is not None:
            self.fight.steps += 1
        self._f9_step()
        if rm.is_over():
            if self.fight is not None:
                self._end_fight(bool(rm.player_won))
        elif rm.phase() != "combat" and self.fight is not None:
            self._end_fight(True)

    def _ooc(self, out: RunDecision) -> E2EDecision | E2ETerminal | Any:
        """A stored OOC row, or (one merged class) the run stepper's next output after
        applying the class."""
        from ppo.encoder import fine_surface

        rm = self.rm
        try:
            dtype, kinds = classify_ooc(out.actions, rm, out.obs)
            keys = ooc_keys(out.actions, kinds, out.obs)
            cls, members = classes_from_keys(keys)
        except C.SchemaError:
            self.counts["unknown"] += 1
            raise
        name = dtype_name(dtype)
        if len(members) == 1:
            self.counts[f"forced/{name}_merged"] += 1
            return self.run.step(members[0][0])
        check_caps(len(out.actions), len(members), name, self.counts)
        fine = fine_surface(out.actions)
        if fine == "unknown":
            self.counts["fine_unknown"] += 1
        sel = C.NO_SELECTION
        if dtype == C.DT["CARD_GRANT"]:
            sel = C.Selection(
                min=int(rm.pending_card_grant_min),
                max=len(rm.card_grant_options()),
                picked=int(self.run.grant_picked),
                can_skip=C.AK["GRANT_CONFIRM"] in kinds,
            )
        event = rm.current_event_name() if dtype in (C.DT["EVENT"], C.DT["NEOW"]) else None
        kw: dict[str, Any] = dict(
            kind=OOC,
            dtype=dtype,
            kinds=kinds,
            actions=out.actions,
            obs_actions=out.actions,
            classes=cls,
            members=members,
            keys=keys,
            sel=sel,
            event_name=event,
            unit=("act", int(rm.act_index)),
            run_obs=out.obs,
            metrics=out.metrics,
            surface=out.surface,
            fine_surface=fine,
        )
        if self.segment.end == "first_ooc":
            self.counts["rows/bootstrap"] += 1
            return self._emit_bootstrap(kw)
        if self.rows >= self.cfg.max_run_decisions:
            return self._cap("run_decisions")
        self.counts[f"fine/{fine}"] += 1
        if len(out.actions) > len(members):
            self.counts["merged_rows"] += 1
            self.counts["merged_actions"] += len(out.actions) - len(members)
        return self._emit(**kw)

    # ── rows and reward ─────────────────────────────────────────────────────

    def _accrue_floors(self) -> None:
        f = int(self.rm.total_floor())
        if f > self.max_floor:
            self._pending_r += R_FLOOR * (f - self.max_floor)
            self.max_floor = f

    def _assign_pending(self) -> float:
        self._accrue_floors()
        r = self._pending_r
        self._pending_r = 0.0
        if self.rows > 0:
            self.rewards[-1] += r
            self.total_reward += r
        else:
            self.unassigned_reward += r
        return r

    def _emit(
        self, *, kind: str, dtype: int, unit: tuple[str, int], count: bool = True, **kw: Any
    ) -> E2EDecision:
        r = self._assign_pending()
        start = unit not in self._units
        if start:
            self._units.add(unit)
        d = E2EDecision(
            row=self.rows,
            kind=kind,
            dtype=dtype,
            reward=r,
            unit=unit,
            unit_start=start,
            act=int(self.rm.act_index),
            floor=int(self.rm.total_floor()),
            rm=self.rm,
            seed=self.seed,
            segment=self.segment,
            **kw,
        )
        self.rewards.append(0.0)
        self.rows += 1
        if count:
            self.counts[f"rows/{dtype_name(dtype)}"] += 1
        self._pending = d
        self.row_meta.append(d.meta())
        return d

    def _emit_bootstrap(self, kw: dict[str, Any]) -> E2EDecision:
        """The segment's last row: value-only, no action, counted as `rows/bootstrap`."""
        d = self._emit(bootstrap=True, count=False, **kw)
        self._done = True
        self.terminal = self._make_terminal("bootstrap", won=False, cap=None, reward=0.0)
        return d

    def _make_terminal(
        self, outcome: str, *, won: bool, cap: str | None, reward: float, error: str | None = None
    ) -> E2ETerminal:
        rm = self.rm
        if outcome == "error":
            try:
                fp = run_fingerprint(rm)
            except BaseException as ex:  # a broken state still ends the segment
                if not ends_segment(ex):
                    raise
                fp = "unavailable"
        else:  # a failure here is a bug: `_drive` turns it into an error terminal
            fp = run_fingerprint(rm)
        return E2ETerminal(
            outcome=outcome,
            won=won,
            cap=cap,
            reward=reward,
            total_reward=self.total_reward,
            rewards=self.rewards,
            unassigned_reward=self.unassigned_reward,
            rows=self.rows,
            steps=int(self.run.steps),
            floor=int(rm.total_floor()),
            act=int(rm.act_index),
            seed=self.seed,
            segment=self.segment,
            bosses=self.bosses,
            reached_boss2=self.reached_boss2,
            fingerprint=fp,
            log_idx=self.log_idx,
            counts=self._final_counts(),
            error=error,
        )

    def _final_counts(self) -> collections.Counter:
        c = collections.Counter(self.counts)
        for k, v in self.run.forced.items():
            c[f"forced/{k}"] += v
        return c

    def _terminal(self, outcome: str, *, won: bool, error: str | None = None) -> E2ETerminal:
        if outcome == "no_actions":
            return self._stall()
        if self.fight is not None:
            self._end_fight(won and bool(self.rm.player_won))
        if outcome == "max_steps":
            return self._cap("steps")
        if won:
            self._pending_r += R_WIN
        r = self._assign_pending()
        self._done = True
        out = "won" if won else ("error" if outcome == "error" else "lost")
        self.counts[f"outcome/{out}"] += 1
        self.terminal = self._make_terminal(out, won=won, cap=None, reward=r, error=error)
        return self.terminal

    def _stall(self) -> E2ETerminal:
        """The run stepper found no legal action in a run that is not over: an engine
        stall, not a death (plan §3 and §6: only deaths and caps are losses). The
        segment ends as an error, `error/NoActions`, which the learner drops and counts,
        like the combat-side stall (GH #1097 ENVR-2)."""
        rm = self.rm
        try:
            raise NoActions(
                f"no legal action at act {rm.act_index}, floor {rm.total_floor()}, "
                f"phase {rm.phase()!r}, step {self.run.steps}"
            )
        except NoActions as ex:
            return self._error(ex)

    def _cap(self, reason: str) -> E2ETerminal:
        if self.fight is not None:
            self.fight = None  # capped: lost, no boss reward
        r = self._assign_pending()
        self._done = True
        self.counts[f"cap/{reason}"] += 1
        self.counts["outcome/capped"] += 1
        self.terminal = self._make_terminal("capped", won=False, cap=reason, reward=r)
        return self.terminal

    # ── fights ──────────────────────────────────────────────────────────────

    def _start_fight(self, cs: Any) -> None:
        from ppo.combat_ctx.intent_log import IntentLog

        factory.check_fight(cs)  # plan §3.1: the fight-start build check
        if self.cfg.tokenizer is not None:
            self.cfg.tokenizer.reset_fight()
        rm = self.rm
        room = fight_room(rm)
        self.fight = _Fight(
            seq=self.fights,
            room=room,
            where=(int(rm.act_index), rm.map_coord()),
            act=int(rm.act_index),
            floor=int(rm.total_floor()),
            ilog=IntentLog(),
        )
        self.fights += 1
        self.counts[f"fights/{room}"] += 1
        if room == "boss" and int(rm.act_index) >= 3:
            self._act3_bosses += 1
            if self._act3_bosses >= 2:
                self.reached_boss2 = True

    def _end_fight(self, won: bool) -> None:
        f = self.fight
        self.fight = None
        if f is None:
            return
        rm = self.rm
        if won:
            self.counts[f"fights_won/{f.room}"] += 1
        if won and f.room == "boss" and not (rm.is_over() and rm.player_won):
            self._pending_r += R_BOSS
            self.bosses += 1

    # ── practice capture (plan §3.3) ────────────────────────────────────────

    def _on_move(self, before: Any, move: dict[str, Any]) -> None:
        rm = self.rm
        room = (rm.current_room() or {}).get("room_type")
        act = int(rm.act_index)
        if act > 3:
            self.counts["capture/act4"] += 1
            return
        if room == "elite":
            cell = f"ELITE_ACT{act}"
        elif room == "boss":
            if act >= 3:
                m = rm.full_map()
                sb = m.get("second_boss")
                cur = m.get("current_coord")
                if sb and cur and (cur["col"], cur["row"]) == (sb["col"], sb["row"]):
                    self.counts["capture/boss2_skipped"] += 1
                    return  # boss 2 has no cell: a double-boss start trains it
            cell = "BOSS_ACT3_DOUBLE" if act == 3 else f"BOSS_ACT{act}"
        else:
            return
        cb = self.cfg.on_capture
        if cb is None:
            return
        self.counts[f"capture/{cell}"] += 1
        cb(
            Capture(
                cell=cell,
                snap=before,
                move=dict(move),
                seed=self.seed,
                steps=int(self.run.steps),
                rows=self.rows,
                idx=tuple(self.log_idx),
            )
        )

    # ── F9 ──────────────────────────────────────────────────────────────────

    def _f9_step(self) -> None:
        """Once per applied step: the checker samples it (`maybe_check`). Legal states the
        checker tolerates (`InvariantChecker.tolerated`) count as `f9/tolerated/<reason>`."""
        chk = self.f9
        calls = chk.calls
        tol = getattr(chk, "tolerated", None)
        before = dict(tol) if tol else None
        found = chk.maybe_check(self.rm, None, run_key=self._f9_key)
        if chk.calls != calls:
            self.counts["f9/checks"] += 1
            if tol:
                for reason, n in tol.items():
                    d = n - (before or {}).get(reason, 0)
                    if d:
                        self.counts["f9/tolerated"] += d
                        self.counts[f"f9/tolerated/{reason}"] += d
        for v in found:
            self.counts["f9/violations"] += 1
            self.counts[f"f9/{v.code}"] += 1
            if len(self.f9_violations) < 8:
                self.f9_violations.append(f"{v.code}: {v.detail}")


def new_env(
    seed: int, *, rm: Any = None, cfg: EnvConfig | None = None, segment: Segment = RUN
) -> Any:
    """`E2EEnv(seed, rm=rm, cfg=cfg, segment=segment)`, or with `cfg.rust` its Rust twin
    (`rsenv.RsE2EEnv`)."""
    if cfg is not None and cfg.rust:
        from .rsenv import RsE2EEnv

        return RsE2EEnv(seed, rm=rm, cfg=cfg, segment=segment, rare=cfg.rust_rare)
    return E2EEnv(seed, rm=rm, cfg=cfg, segment=segment)


# ── the env's part of the row batch (contracts.ROW_FIELDS) ───────────────────


def segment_rows(
    rows: list[E2EDecision] | list[RowMeta], term: E2ETerminal
) -> dict[str, np.ndarray]:
    """The env's fields of the row batch for one finished segment, in contracts dtypes.

    `rows` are the segment's decisions or their `RowMeta` (`E2EEnv.row_meta`; or call
    `E2EEnv.segment_rows()`), all of them, in order.

    Per row: `run_uid`, `t`, `is_combat`, `dtype`, `reward` (all reward after the
    decision, auto-advanced steps included: `term.rewards`), `done` (the segment ended
    after it: the last row of every finished segment), `bootstrap`, `n_classes`, `n_mem`
    and `cell`; ragged by `n_mem`, `mem_class`; by `n_classes`, `cls_kind`. The actor adds
    the rest (actor, logp_behav, action, iter, v_behav, tokens, teacher probabilities).

    The last row ends the segment one of two ways (SCHEMA.md §9.1): a terminal (won, lost
    or capped; `done`, not `bootstrap`; its `reward` includes the outcome) or a practice
    end (the first OOC decision of a practice segment; `done` and `bootstrap`, reward 0).
    An iteration cut (`bootstrap`, not `done`) is the learner's, never the env's. An
    `error` segment also ends `done`; the actor should drop it.
    """
    rows = [r.meta() if isinstance(r, E2EDecision) else r for r in rows]
    if len(rows) != term.rows or [r.t for r in rows] != list(range(term.rows)):
        raise ValueError(f"segment_rows needs all {term.rows} rows of the segment, in order")
    n = len(rows)

    def col(name: str, vals: list[Any]) -> np.ndarray:
        return np.asarray(vals, dtype=np.dtype(C.FIELDS[name].dtype))

    out = {
        "run_uid": col("run_uid", [r.run_uid for r in rows]),
        "t": col("t", [r.t for r in rows]),
        "is_combat": col("is_combat", [r.is_combat for r in rows]),
        "dtype": col("dtype", [r.dtype for r in rows]),
        "reward": col("reward", list(term.rewards)),
        "done": col("done", [i == n - 1 for i in range(n)]),
        "bootstrap": col("bootstrap", [r.bootstrap for r in rows]),
        "n_classes": col("n_classes", [r.n_classes for r in rows]),
        "n_mem": col("n_mem", [r.n_mem for r in rows]),
        "cell": col("cell", [r.cell for r in rows]),
        "mem_class": col("mem_class", [c for r in rows for c in r.classes]),
        "cls_kind": col("cls_kind", [k for r in rows for k in r.cls_kind]),
    }
    return out


def _combat_to_rm(a: dict[str, Any]) -> dict[str, Any]:
    from ppo.combat_bench.oracle import combat_to_rm

    return combat_to_rm(a)
