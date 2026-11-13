"""GH #399: the env must fail loudly on action kinds the encoder cannot encode.

`encoder._featurize_action` ends in `_fixed_tok("none")` — an unknown action
kind becomes a *blank* token rather than a crash (encoder.py:390-391). That
fallback is deliberate and load-bearing (the encoder is the porting authority
for the bit-for-bit Rust mirror; it must not learn to panic), but it means a
newly-reachable surface arrives as a list of identical blank tokens and the
policy ranks pure noise while every dashboard stays green. This is the same
invisibility family as #317 (events auto-resolved, `emb_event` drift 0.000000)
and the `auto_front` single-option masquerade of Task 1.

The deployed Rust arm already refuses to encode such a list — `PpoPolicy`
short-circuits a `PotionSwap`-headed list to the heuristic
(`sts2-runai/src/ppo/policy.rs:764-772`) precisely because there is no
`ActionDesc` tag for it. Python had no equivalent. These tests pin the guard
at the env boundary: every non-combat decision the training loop touches is
checked against `encoder._KIND_SURFACE` before it can reach the encoder.

`interactive_potion_swap` is the concrete instance (deferred to #379): it is
the one RunManager flag with zero Python encoder support, so `potion_swap` is
used here as the canonical unencodable kind.
"""

from __future__ import annotations

import os
import sys

import pytest
import sts2_client

PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))


@pytest.fixture(autouse=True)
def _syspath(monkeypatch):
    monkeypatch.syspath_prepend(PY)


class _StubRm:
    """Duck-typed RunManager: enough surface for `_advance` + `metrics_snapshot`.

    `sts2_rs.RunManager` is a PyO3 native type whose methods cannot be
    monkeypatched, so the decision loop is driven over a stub instead (the
    env holds `self.rm` as a plain attribute).
    """

    def __init__(self, phase: str, actions: list[dict]) -> None:
        self._phase = phase
        self._actions = actions
        self.act_index = 0
        self.ascension_level = 0
        self.current_hp = 70
        self.max_hp = 70
        self.gold = 99
        self.player_won = False

    def is_over(self) -> bool:
        return False

    def phase(self) -> str:
        return self._phase

    def available_actions(self) -> list[dict]:
        return list(self._actions)

    def materialize_event(self) -> None:  # pragma: no cover - phase != event here
        pass

    # metrics_snapshot / mask_rejectable surface
    def master_deck_detailed(self):
        return [("Strike", False)]

    def total_floor(self):
        return 1

    def act_floor(self):
        return 1

    def relics(self):
        return []

    def potions(self):
        return []

    def max_potion_slots(self):
        return 3


class _HaltAgent:
    """Combat agent that immediately halts (returns None from `step`)."""

    def step(self, rm):
        return None


def _env_over(phase: str, actions: list[dict]):
    """A policy-mode env whose RunManager is the stub above."""
    from ppo.env import PpoRunEnv

    e = PpoRunEnv(seed=1)
    e.rm = _StubRm(phase, actions)
    e.agent = _HaltAgent()
    return e


POTION_SWAP = [
    {"action": "potion_swap", "discard_slot": None},
    {"action": "potion_swap", "discard_slot": 0},
]

# A kind no engine emits and no handler intercepts. `potion_swap` used to play
# this role, but GH #379 gave it a real encoder surface, so it is now a
# POSITIVE case (see `test_potion_swap_is_no_longer_the_unencodable_case`) and
# can no longer exercise the env boundary. A synthetic kind cannot be quietly
# made reachable the same way.
UNENCODABLE = [
    {"action": "unencodable_probe", "slot": None},
    {"action": "unencodable_probe", "slot": 0},
]


def test_unencodable_action_kind_raises() -> None:
    """A multi-option list of an unencodable kind must not reach the encoder."""
    e = _env_over("card_reward", UNENCODABLE)
    with pytest.raises(RuntimeError, match="unencodable_probe"):
        e.reset()


def test_guard_message_names_the_failure_mode() -> None:
    """The message has to say what silently happens otherwise, and where it's tracked."""
    from ppo.env import assert_encodable

    with pytest.raises(RuntimeError) as excinfo:
        assert_encodable(UNENCODABLE, phase="card_reward")
    msg = str(excinfo.value)
    assert "unencodable_probe" in msg
    assert "blank" in msg
    assert "#399" in msg and "#317" in msg


def test_single_option_unencodable_also_raises() -> None:
    """Auto-take must not be an escape hatch.

    A lone unencodable action is auto-resolved without ever being featurized,
    so no blank token is emitted — but it is exactly the #317 shape (a real
    surface resolved at option 0 while the net never learns it exists), and
    the flags that make it reachable are the ones this campaign flips.
    """
    e = _env_over("card_reward", [{"action": "unencodable_probe", "slot": None}])
    with pytest.raises(RuntimeError, match="unencodable_probe"):
        e.reset()


def test_known_kinds_do_not_raise() -> None:
    """Every currently-reachable non-combat kind is in `_KIND_SURFACE`."""
    from ppo.encoder import _KIND_SURFACE
    from ppo.env import assert_encodable

    assert_encodable([{"action": k} for k in _KIND_SURFACE], phase="card_reward")


def test_deck_mutation_kinds_are_encodable() -> None:
    """The four flags Task 3 flips must not trip the guard they precede."""
    from ppo.env import assert_encodable

    assert_encodable(
        [
            {"action": "remove_deck_card", "deck_index": 0},
            {"action": "upgrade_deck_card", "deck_index": 1},
            {"action": "transform_deck_card", "deck_index": 2},
            {"action": "enchant_deck_card", "deck_index": 3},
            {"action": "grant_deck_card", "card_id": "Strike"},
        ],
        phase="event",
    )


def test_combat_actions_bypass_the_guard() -> None:
    """The guard sits *after* the combat branch.

    Combat is owned by the MCTS agent and its action kinds (`play_card`,
    `end_turn`, `drink_potion`, …) are deliberately absent from
    `_KIND_SURFACE` — the run encoder never sees them. A guard placed before
    the combat dispatch would fire on every single combat turn.
    """
    from ppo.env import Terminal

    e = _env_over("combat", [{"action": "play_card", "card_index": 0}, {"action": "end_turn"}])
    out = e.reset()
    assert isinstance(out, Terminal)
    assert out.outcome == "agent_halt"


# ── GH #399 Task 3: the four flags are actually ON and reach the policy ──────

# Seed cover for the four deck-mutation kinds under the *cheap* combat budget
# used below (`sims=8`): removal on 3/8, upgrade on 5/17, transform on 8/15,
# enchant on 5. These surfaces are seed-dependent (they come from Neow
# blessings, events like Wellspring / Wood Carvings, and on-obtain relics such
# as Precise Scissors / Pomander), so a cover is the only way to see all four
# without driving dozens of runs. If an engine change moves the cover this
# test fails loudly — re-derive it by censusing kinds over seeds 0..40 rather
# than by deleting the assertion: "no removal ever surfaced" is exactly the
# invisibility bug #399 exists to close.
#
# The cover is per game build: v0.111.0 rehashed every seed (XxHash64), so it
# was re-derived there by that census (seeds 0..40). The first cover, 14 + 16,
# went stale when seed 16 stopped surfacing its removal (already so on main at
# 2bc55fb2, before #1181's changes). Re-derived 2026-11-08 (#1181): removal on
# 11/12, upgrade on 9/16/17/18/36/39, transform on 0/4/12/14/15/18/…, enchant
# on 1/3/14/17/22/37. The cover below has every kind on TWO seeds, so one
# seed drifting no longer turns the gate red; all of a kind's seeds drifting
# still does.
#
# v0.113.0: same census on the v0.113.0 engine (2026-11-13). Removal on
# 6/11/16/29, upgrade on 9/16/18, transform on 0/4/14/15/18/19/30/38, enchant
# on 1/3/14/17/21/22/24/37; seed 12 surfaces nothing there, so the v0.111.0
# cover would lose its second removal. Re-derive it once the v0.113.0 run
# level lands (GH #1303): that port moves these seeds.
_COVER_SEEDS_BY_VERSION = {
    "v0.107.1": (3, 5, 8, 15, 17),
    "v0.111.0": (11, 12, 17, 18, 22),
    "v0.113.0": (11, 14, 16, 17, 18),
}
_COVER_SEEDS = _COVER_SEEDS_BY_VERSION[sts2_client.GAME_VERSION]
_DECK_MUTATION_KINDS = frozenset(
    {"remove_deck_card", "upgrade_deck_card", "transform_deck_card", "enchant_deck_card"}
)
# Anything the engine can preempt the whole action list with. `_advance`
# dispatches on `rm.phase()`, so one of these surfacing while the phase is
# still "combat" would be routed to the combat MCTS and never reach the net.
_PREEMPT_KINDS = _DECK_MUTATION_KINDS | {"grant_deck_card", "potion_swap"}


class _CombatPreemptProbe:
    """Transparent proxy over the env's Rust agent.

    Records any step where the engine offered a preempt list while the phase
    was still `combat` — see `test_deck_mutations_never_preempt_during_combat`.
    """

    def __init__(self, inner, hits: list) -> None:
        self._inner = inner
        self._hits = hits

    def step(self, rm):
        if rm.phase() == "combat":
            actions = rm.available_actions()
            if actions and actions[0]["action"] in _PREEMPT_KINDS:
                self._hits.append((int(rm.total_floor()), actions[0]["action"]))
        return self._inner.step(rm)


def _expert_census(seed: int) -> tuple[dict, list]:
    """Drive one cheap expert run; return {kind: [Decision, ...]} and preempt hits."""
    # The module-scoped fixture below is built before the function-scoped
    # `_syspath` monkeypatch runs, so make the import self-sufficient.
    if PY not in sys.path:
        sys.path.insert(0, PY)
    from ppo.env import PpoRunEnv, Terminal

    env = PpoRunEnv(seed=seed, expert=True, sims=8, boss_sims=8)
    hits: list = []
    env.agent = _CombatPreemptProbe(env.agent, hits)
    by_kind: dict = {}
    for out in env.expert_run():
        if isinstance(out, Terminal):
            break
        by_kind.setdefault(out.actions[0]["action"], []).append(out)
    return by_kind, hits


@pytest.fixture(scope="module")
def _cover_census():
    """One cheap expert run per cover seed, shared by the three tests below."""
    return {seed: _expert_census(seed) for seed in _COVER_SEEDS}


def test_deck_mutation_decisions_actually_reach_the_policy(_cover_census) -> None:
    """All four flags must be ON *and* produce yielded decisions.

    This is the campaign's premise, asserted rather than assumed: before #399
    the engine auto-resolved every removal (front of deck), upgrade (random),
    transform and enchant, so the run policy had never made one of these
    picks. A zero count for any kind means a flag did not reach the surface.
    """
    seen = set()
    for by_kind, _ in _cover_census.values():
        seen |= _DECK_MUTATION_KINDS & set(by_kind)
    assert seen == _DECK_MUTATION_KINDS, (
        f"missing deck-mutation kinds: {_DECK_MUTATION_KINDS - seen}"
    )


def test_deck_mutation_decisions_are_card_picks_resolved_in_range(_cover_census) -> None:
    """Every deck-mutation decision is a `card-pick` the expert answers in range.

    The expert here IS the Rust `HeuristicRunAgent` — the same
    `MctsCombatPolicy<HeuristicPolicy>` `run_arena --policy heuristic` builds
    — so there is no second Python heuristic to keep in sync; G1 pins the two
    per seed. What can still break is the index space: the engine offers
    `idx` over an *eligible subset* (curse-front removal order, upgradable
    cards, transform-eligible cards), so an out-of-range or unanswered pick
    means the policy fell through to a phase branch instead of
    `choose_deck_mutation`.
    """
    checked = 0
    for seed, (by_kind, _) in _cover_census.items():
        for kind in _DECK_MUTATION_KINDS:
            for dec in by_kind.get(kind, ()):
                assert dec.surface == "card-pick", f"seed {seed}: {kind} surfaced as {dec.surface}"
                assert dec.expert_idx is not None, f"seed {seed}: {kind} left unresolved"
                assert 0 <= dec.expert_idx < len(dec.actions), (
                    f"seed {seed}: {kind} idx {dec.expert_idx} out of range"
                )
                checked += 1
    assert checked, "vacuous — no deck-mutation decision was inspected"


def test_deck_mutations_never_preempt_during_combat(_cover_census) -> None:
    """A preempt list must never surface while `rm.phase() == 'combat'`.

    `RunManager::get_available_actions` checks the deck-mutation / card-grant
    / potion-swap queues *before* the `match self.phase` arm
    (`sts2-core/src/run_manager.rs:934-1017`), while `_advance` dispatches on
    the phase — so a queue filled mid-combat would hand a card pick to the
    combat MCTS, bypassing both `assert_encodable` and the net. It cannot
    happen today (no combat-side code touches those queues, and the one
    run-side path that fills them while `self.phase` is transiently `Combat`
    — `route_after_event_rewards` reached via `finish_event_combat` — always
    reassigns the phase before returning). This pins that.
    """
    hits = {seed: h for seed, (_, h) in _cover_census.items() if h}
    assert not hits, f"deck-mutation preempt surfaced during combat: {hits}"


# ── GH #399 Task 4: the deck-mutation decisions must be COUNTABLE ────────────

# Playbook §0.1's pre-flight reads `dec_per_run/<surface>` out of
# `<save>.stats.jsonl` and aborts a campaign whose new surface reads ~0. Every
# deck-mutation decision classifies as the COARSE surface `card-pick`
# (`env.classify_surface`), i.e. the same bucket as the 14-per-run card
# rewards — a removal count of 0 would be invisible inside it. The counters
# therefore carry the fine surface (`encoder.fine_surface`) alongside the
# coarse one.
#
# Both families are emitted, and the COARSE count wins a name collision:
# `dec_per_run/event` is the sentinel #317 left behind (1.2 = broken, 6.x =
# healthy — see the playbook and the it150/it180 provenance notes), so it must
# keep meaning "decisions taken in the event phase" even for an event whose
# candidates are e.g. relic picks. The fine keys are an overlay on top of the
# coarse partition, not a replacement: summing all `dec_per_run/*` double
# counts, by design.

_FINE_DECK_MUTATION_SURFACES = (
    "card-remove",
    "card-upgrade",
    "card-transform",
    "card-enchant",
    "card-grant",
)


def test_behavior_stats_splits_card_pick_by_fine_surface() -> None:
    """The four (five, with grants) deck-mutation surfaces get their own key."""
    from ppo.trainer import behavior_stats

    trans = [{"surface": "card-pick", "fine_surface": f} for f in _FINE_DECK_MUTATION_SURFACES]
    trans += [
        {"surface": "card-pick", "fine_surface": "card-reward"},
        {"surface": "card-pick", "fine_surface": "card-reward"},
        {"surface": "event", "fine_surface": "event"},
    ]
    st = behavior_stats([{"transitions": trans}])
    for fine in _FINE_DECK_MUTATION_SURFACES:
        assert st[f"dec_per_run/{fine}"] == pytest.approx(1.0), st
    assert st["dec_per_run/card-reward"] == pytest.approx(2.0)
    # Coarse family untouched: the card-pick bucket is still the whole bucket.
    assert st["dec_per_run/card-pick"] == pytest.approx(7.0)
    assert st["dec_per_run/event"] == pytest.approx(1.0)


def test_coarse_event_counter_keeps_its_meaning() -> None:
    """`dec_per_run/event` stays coarse-defined — it is the §0.1 sentinel.

    A fine surface can refine `event` too (an event offering relic picks is
    coarse `event`, fine `relic-reward`). Re-keying wholesale would silently
    shrink the one series every campaign log and provenance note compares
    against; the coarse count therefore wins the name collision.
    """
    from ppo.trainer import behavior_stats

    st = behavior_stats(
        [
            {
                "transitions": [
                    {"surface": "event", "fine_surface": "relic-reward"},
                    {"surface": "event", "fine_surface": "event"},
                ]
            }
        ]
    )
    assert st["dec_per_run/event"] == pytest.approx(2.0)
    assert st["dec_per_run/relic-reward"] == pytest.approx(1.0)


def test_behavior_stats_tolerates_transitions_without_a_fine_surface() -> None:
    """Rows collected before this change (or by another producer) still count."""
    from ppo.trainer import behavior_stats

    st = behavior_stats([{"transitions": [{"surface": "card-pick"}, {"surface": "path"}]}])
    assert st["dec_per_run/card-pick"] == pytest.approx(1.0)
    assert st["dec_per_run/path"] == pytest.approx(1.0)


def test_behavior_stats_averages_over_runs() -> None:
    """Counters are per-run means, not totals — two runs, one removal each."""
    from ppo.trainer import behavior_stats

    row = {"transitions": [{"surface": "card-pick", "fine_surface": "card-remove"}]}
    st = behavior_stats([row, row])
    assert st["dec_per_run/card-remove"] == pytest.approx(1.0)
    assert behavior_stats([row, {"transitions": []}])["dec_per_run/card-remove"] == pytest.approx(
        0.5
    )


def test_deck_mutation_decisions_carry_a_distinct_fine_surface(_cover_census) -> None:
    """End of the chain, against the real engine: the census decisions map to
    distinct fine surfaces even though all five share the coarse `card-pick`."""
    from ppo.encoder import fine_surface

    seen = set()
    for by_kind, _ in _cover_census.values():
        for kind in _DECK_MUTATION_KINDS:
            for dec in by_kind.get(kind, ()):
                assert dec.surface == "card-pick"
                seen.add(fine_surface(dec.actions))
    assert seen == {"card-remove", "card-upgrade", "card-transform", "card-enchant"}, seen


def test_worker_transitions_carry_the_fine_surface(monkeypatch) -> None:
    """Wiring gate: the trainer only ever sees what `play_one` records.

    Driven with a stub net (uniform sampling over the real candidates, so no
    greedy shop deadlock) over the real env, so this pins the whole chain
    env → `play_one` transitions → `behavior_stats`.
    """
    from ppo import workers
    from ppo.encoder import fine_surface
    from ppo.trainer import behavior_stats

    class _StubPolicy:
        def sample(self, obs, actions, metrics, rng, greedy=False):
            return int(rng.integers(len(actions))), 0.0, 0.0, {}

    monkeypatch.setattr(workers, "_net_policy", lambda ckpt_path: _StubPolicy())
    row = workers.play_one(seed=3, policy="net", record_decisions=True, sims=8, boss_sims=8)
    trans, decs = row["transitions"], row["decisions"]
    assert trans and len(trans) == len(decs)
    for t, d in zip(trans, decs, strict=True):
        assert t["surface"] == d["surface"]
        assert t["fine_surface"] == fine_surface(d["actions"])
    st = behavior_stats([row])
    assert st[f"dec_per_run/{trans[0]['fine_surface']}"] > 0


def test_potion_swap_is_no_longer_the_unencodable_case() -> None:
    """GH #379 flipped `potion_swap` from the canonical negative to a positive.

    Asserted here, in the guard's own file, so that re-adding it to the
    negative list above fails loudly rather than quietly re-opening the
    invisibility bug from the other side.
    """
    from ppo.encoder import _KIND_SURFACE
    from ppo.env import assert_encodable

    assert _KIND_SURFACE["potion_swap"] == "potion-swap"
    assert_encodable(
        [
            {"action": "potion_swap", "discard_slot": None},
            {"action": "potion_swap", "discard_slot": 0},
        ],
        phase="card_reward",
    )


def test_potion_swap_is_intercepted_not_faulted(monkeypatch, tmp_path) -> None:
    """With an OC table loaded, `potion_swap` is resolved rule-side.

    That is the #479 measurement arm: `_advance` routes the whole set to the
    combat agent, whose Rust `MctsCombatPolicy::choose` applies the same OC
    rule as the deployed arena, and the interception sits *before*
    `assert_encodable`. Without a table the set falls through to the encoder
    instead — GH #379 gave it a real surface — which is what
    `test_potion_swap_is_no_longer_the_unencodable_case` above pins.
    """
    # The OC seam loads the table eagerly and refuses to run alongside the
    # leaf potion term (it would double-count the value the root charge
    # prices), so a real file and both vars are part of turning the arm on.
    table = tmp_path / "oc.json"
    table.write_text(
        '{"version":1,"character":"ironclad","units":"hp_equiv",'
        '"source":"manual","default_oc":8.0,"entries":[]}'
    )
    monkeypatch.setenv("STS2_RUNAI_POTION_OC", str(table))
    monkeypatch.setenv("STS2_RUNAI_POTION_WEIGHT", "0")

    class _OneShotRm(_StubRm):
        """Serves the swap set once, then reports the run over."""

        def __init__(self) -> None:
            super().__init__("card_reward", POTION_SWAP)
            self.served = 0

        def is_over(self) -> bool:
            return self.served > 0

        def available_actions(self) -> list[dict]:
            self.served += 1
            return list(self._actions)

    class _ApplyAgent:
        """Combat agent that resolves the swap (returns a non-None step)."""

        def step(self, rm):
            return {}

    from ppo.env import PpoRunEnv

    e = PpoRunEnv(seed=1)
    e.rm = _OneShotRm()
    e.agent = _ApplyAgent()
    e.reset()  # must not raise the #399 guard
    assert [k for *_, k in e.resolution_log] == ["potion-swap-rule"]
