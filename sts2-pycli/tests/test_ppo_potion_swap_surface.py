"""GH #379: the full-belt potion swap must be a real encoder surface.

Before this, ``potion_swap`` had no entry in ``encoder._KIND_SURFACE``, so
``env.assert_encodable`` refused the list and ``interactive_potion_swap`` had
to stay pinned OFF on both arms — a full-belt potion reward simply evaporated
into ``potions_overflow_dropped``. The deployed Rust arm short-circuited the
same list to ``HeuristicPolicy`` (potion-rarity ladder), so the net never saw
the decision on either side.

What the tokenization has to get right, and what each test here pins:

1. **Direction.** Each option names the potion you would THROW AWAY, not the
   one you would gain — ``discard_slot`` indexes the belt. Getting this
   backwards would train the net to reason about the wrong potion entirely.
2. **The offer.** What you gain is identical across every option, so it rides
   on the STATE token (``state_cat[3]``). Without it the decision is
   unanswerable: every option would look like a pure loss.
3. **Inertness elsewhere.** ``state_cat[3]`` must stay the potion pad row on
   every other surface, or this change would perturb the observation of every
   already-trained surface and invalidate the champion for free.

Deliberately NOT routed through ``ppo_encoder_parity.jsonl``: at the measured
0.075 full-belt potion rewards per Ironclad A0 run, requiring a fixture row
would cost the dump ~13 extra full runs. ``dump_parity_fixtures`` keeps such a
row opportunistically; this pins the branch outright. The Rust twin lives in
``sts2-runai/src/ppo/encoder.rs::potion_swap_tokens``.
"""

from __future__ import annotations

import pytest
from ppo import encoder, env, vocab

# Two potions on the belt and a third being offered. Registry spellings are
# `RunManager.potions()`'s debug names (`format!("{k:?}")`), the same ones
# `pending_potion_swap` returns — see `sts2-py/src/run_manager.rs`.
HELD = ["BlockPotion", "FirePotion"]
OFFERED = "EnergyPotion"


def _obs(pending: str | None = OFFERED) -> dict:
    o = {
        "phase": "card_reward",
        "surface": "potion-swap",
        "deck": [["CARD.STRIKE_IRONCLAD", False]],
        "relics": [],
        "potions": list(HELD),
        "map": None,
        "map_coord": None,
        "max_hp": 80,
        "ascension": 0,
    }
    if pending is not None:
        o["pending_potion_swap"] = pending
    return o


def _actions() -> list[dict]:
    """The engine's own order (`run_manager.rs:1011`): decline, then one
    discard option per held slot."""
    return [
        {"action": "potion_swap", "discard_slot": None},
        *({"action": "potion_swap", "discard_slot": i} for i in range(len(HELD))),
    ]


def test_surface_is_claimed_before_the_phase():
    """A swap preempts the action list in ANY phase, so the phase it happens
    to interrupt must not name the surface."""
    assert env.classify_surface("card_reward", _actions()) == "potion-swap"
    assert env.classify_surface("map_choice", _actions()) == "potion-swap"
    assert encoder.fine_surface(_actions()) == "potion-swap"


def test_list_is_encodable():
    """The guard that kept the flag off. It must now pass."""
    env.assert_encodable(_actions(), phase="card_reward")


def test_option_tokens_name_the_discarded_potion():
    feat = encoder.featurize_row(_obs(), _actions(), {})
    act_cat = feat.act_cat.tolist()
    assert len(act_cat) == 3

    # Option 0 = decline: a fixed token, not a potion.
    assert act_cat[0][6] == vocab.fixed_index("decline_potion")
    assert act_cat[0][3] == encoder.empty_cat()[3], "decline names no potion"

    # Options 1..n = the held potion at that slot, in belt order.
    for slot, pid in enumerate(HELD):
        assert act_cat[slot + 1][3] == vocab.potions().index(pid), f"slot {slot}"
        assert act_cat[slot + 1][6] == encoder.empty_cat()[6], "not a fixed token"

    # The two discard options must be DISTINGUISHABLE — a token that collapsed
    # both to the same vector would leave the net picking by position.
    assert act_cat[1][3] != act_cat[2][3]


def test_state_token_carries_the_offered_potion():
    feat = encoder.featurize_row(_obs(), _actions(), {})
    state_cat = feat.obs_cat.tolist()[0]
    assert state_cat[8] == vocab.surface_index("potion-swap")
    assert state_cat[3] == vocab.potions().index(OFFERED)
    # And it is NOT one of the held ones, i.e. the offer really did come from
    # `pending_potion_swap` rather than leaking off the belt.
    assert state_cat[3] not in {vocab.potions().index(p) for p in HELD}


@pytest.mark.parametrize("pending", [None, ""])
def test_state_token_pads_when_no_swap_is_pending(pending):
    """The inertness contract: on every surface that is not a potion swap,
    `state_cat[3]` stays the pad row, so this change cannot perturb an
    already-trained surface's observation."""
    obs = _obs(pending=pending)
    feat = encoder.featurize_row(obs, _actions(), {})
    assert feat.obs_cat.tolist()[0][3] == encoder.empty_cat()[3]


def test_ordinary_surface_is_unperturbed():
    """A card reward encodes byte-identically whether or not this code exists
    — `pending_potion_swap` is absent from its obs, so the state token's
    potion slot stays padded."""
    obs = {
        "phase": "card_reward",
        "surface": "card-pick",
        "deck": [["CARD.STRIKE_IRONCLAD", False]],
        "relics": [],
        "potions": list(HELD),
        "map": None,
        "map_coord": None,
        "max_hp": 80,
        "ascension": 0,
        "card_reward_option_ids": ["CARD.BASH", "CARD.CLEAVE"],
    }
    actions = [
        {"action": "pick_card_reward", "idx": 0},
        {"action": "pick_card_reward", "idx": 1},
    ]
    state_cat = encoder.featurize_row(obs, actions, {}).obs_cat.tolist()[0]
    assert state_cat[3] == encoder.empty_cat()[3]
    assert state_cat[8] == vocab.surface_index("card-reward")


def test_appending_kept_every_pre_existing_embedding_row_in_place():
    """`model.load_compat` warm-starts an older checkpoint by copying rows
    0..N verbatim, so a new vocab entry is only safe if it is APPENDED. If
    someone inserts one mid-table, every trained row after it silently shifts
    meaning — this catches that at the vocab, before a campaign burns on it."""
    assert vocab.SURFACES[-1] == "potion-swap"
    assert vocab.SURFACES.index("unknown") == 14, "pre-existing rows must not move"
    assert vocab.FIXED_KINDS[-1] == "decline_potion"
    assert vocab.FIXED_KINDS.index("sacrifice") == 10, "pre-existing rows must not move"
    # The fallback resolves by NAME; "unknown" is no longer the last entry, so
    # a positional fallback would now hand every unrecognized surface to
    # "potion-swap".
    assert vocab.surface_index("no-such-surface") == vocab.surface_index("unknown")


def test_pad_indices_did_not_move():
    """The load-bearing one, and the reason the new entries went into the
    SPARE row rather than growing the table.

    A torch checkpoint tolerates a grown embedding (``model.load_compat``
    copies rows 0..N and zero-fills). An exported **ONNX** champion does not
    — its tables are baked at export size, so moving a pad index feeds every
    deployed net an out-of-range gather. Moving them also perturbs the
    featurization of every *already-trained* surface, since the pad index
    appears in `empty_cat()` and therefore in almost every token: the encoder
    parity fixture (``ppo_encoder_parity.jsonl``) failed on exactly that when
    these were first written as ``len(...) + 1``.
    """
    from ppo import schema

    pads = schema.pad_indices()
    assert pads["surface"] == 16, "surface pad moved — every .onnx needs re-export"
    assert pads["fixed"] == 12, "fixed pad moved — every .onnx needs re-export"
    # The spare is now spent. Whoever adds the next entry gets a red test
    # here, which is the point: they need to know it costs a re-export.
    assert len(vocab.SURFACES) == vocab.N_SURFACE, "no spare surface row left"
    assert len(vocab.FIXED_KINDS) == vocab.N_FIXED, "no spare fixed row left"


COVER_SEEDS = (2, 6, 14, 22)


@pytest.mark.slow
def test_live_path_carries_the_offer_into_the_state_token(monkeypatch):
    """End-to-end over the real engine — the one thing the tests above cannot
    prove.

    Every assertion in this file so far runs on a hand-built obs dict, so all
    of them stay green if the *plumbing* breaks: an obs key renamed,
    `observe()`'s surface arm dropped, `classify_surface` reordered so the
    swap is reported as whatever room it interrupted. Any of those leaves
    `pending_potion_swap` absent, `state_cat[3]` falls back to the potion pad
    row, and the net silently loses the only signal telling it what it would
    GAIN — the exact defect this surface exists to fix, and one that fails
    open rather than loud.

    `interactive_potion_swap` is OFF unless `STS2_RUNAI_POTION_SWAP=1`
    (`potion_oc::resolve_potion_swap_flag`: "Unset / `0` => off"), and
    `PpoRunEnv` takes it from that same resolver, so the surface simply does
    not occur in a default environment — a census over seeds 0..20 finds zero
    swaps. Set it here rather than assuming an ambient export, which is how
    this test could pass for its author and fail everywhere else.

    `STS2_RUNAI_POTION_OC` must stay UNSET: `potion_oc::swap_is_rule_side` is
    `oc_env.is_some()`, so an OC table would resolve the swap rule-side and
    never yield it as a decision — the exact set this test needs to inspect.

    `COVER_SEEDS` are known cover seeds: with the flag on, each produces at
    least one full-belt potion reward at A0. Re-censused 2026-11-14 over
    0..39: 2, 6, 14, 22, 26, 39 swap. The 2026-11-12 set (GH #1301: 5, 12,
    13, 15, 27, 31, 36, 38) was empty on main at 5a96d6647, after v0.111.0
    became the default build (#1358). Seed 17, the
    previous single cover seed, had lost its swap by then (most likely to
    act 1 rolling from the seed, `7320dbe99`); 15 and 5 had each drifted
    out earlier and drifted back in. A
    single seed broke on every engine change that touched the route, so
    the test now takes the first seed of a small set that surfaces a swap.
    If an engine change empties the whole set, the test fails loudly —
    re-derive it by censusing `potion_swap` over seeds 0..39 rather than
    deleting the assertion, because "no swap ever surfaced" is precisely
    the invisibility bug being closed here.

    The test is `slow`, so `just test` skips it; `just test-slow` and
    `just test-public-shape` (`-m "not oracle"`) run it.
    """
    monkeypatch.setenv("STS2_RUNAI_POTION_SWAP", "1")
    monkeypatch.delenv("STS2_RUNAI_POTION_OC", raising=False)

    out = next((o for seed in COVER_SEEDS if (o := _first_potion_swap(seed))), None)
    if out is None:
        pytest.fail(f"no cover seed in {COVER_SEEDS} produces a potion swap — re-derive them")

    offered = out.obs.get("pending_potion_swap")
    belt = out.obs.get("potions")
    assert offered, "obs carries no offered potion on the live path"
    assert out.surface == "potion-swap"

    feat = encoder.featurize_row(out.obs, out.actions, out.metrics)
    state_cat = feat.obs_cat.tolist()[0]
    assert state_cat[8] == vocab.surface_index("potion-swap")
    assert state_cat[3] == vocab.potions().index(offered)
    assert state_cat[3] != encoder.empty_cat()[3], "offer fell back to the pad row"

    for action, cat in zip(out.actions, feat.act_cat.tolist(), strict=True):
        slot = action.get("discard_slot")
        if slot is None:
            assert cat[6] == vocab.fixed_index("decline_potion")
        else:
            assert cat[3] == vocab.potions().index(belt[slot]), f"slot {slot}"


def _first_potion_swap(seed):
    """The first `potion_swap` decision of an expert run on `seed`, or None."""
    from ppo.env import PpoRunEnv, Terminal

    for out in PpoRunEnv(seed=seed, expert=True).expert_run():
        if isinstance(out, Terminal):
            return None
        if out.actions and out.actions[0]["action"] == "potion_swap":
            return out
    return None


# --- the startup banner (GH #545) -------------------------------------------
#
# The tests above pin what the surface DOES once it is on. These pin that a
# campaign can tell whether it is on at all. That gap is not hypothetical: the
# #545 v2 campaign dropped `export STS2_RUNAI_POTION_SWAP=1` from its launcher
# and trained 200 iterations with full-belt potion rewards silently dropped.
# Its pre-flight check read `dec_per_run/potion-swap` — and `behavior_stats`
# builds that from a `Counter` over OBSERVED surfaces, so a surface that never
# fires emits no key at all. The check tested for a zero, an absent key is not
# a zero, and it passed vacuously for the entire campaign.


@pytest.mark.parametrize(
    ("env_value", "expect_on"),
    [("1", True), ("0", False), (None, False)],
)
def test_the_banner_states_the_flag_in_every_state(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None, expect_on: bool
) -> None:
    """Printed ON *and* off — a deviation-only banner would be just as silent.

    Every other `SCREEN:` line in `trainer.py` fires only when something is
    switched on, which is right when the risk is an unwanted setting. Here the
    default IS the dangerous state (the recipe wants the flag on), so the
    unset case is the one that must announce itself. Parametrised over all
    three env values so a regression to `if on:` fails here rather than in a
    31-hour campaign log.
    """
    from ppo.trainer import potion_swap_banner

    if env_value is None:
        monkeypatch.delenv("STS2_RUNAI_POTION_SWAP", raising=False)
    else:
        monkeypatch.setenv("STS2_RUNAI_POTION_SWAP", env_value)

    line = potion_swap_banner()

    assert line, "the banner is empty — nothing announces the flag"
    assert "STS2_RUNAI_POTION_SWAP" in line, (
        "the banner must name the variable, so a reader who sees the wrong state knows what to set"
    )
    # Match the whole `potion_swap=<state>` token, never a bare "ON": the
    # substring "ON" also occurs inside "STS2_RUNAI_POTION_SWAP" (POTI-ON),
    # so `"ON" in line` is True even when the flag is off. That is a check
    # that cannot come out differently when the claim is false.
    assert f"potion_swap={'ON' if expect_on else 'off'}" in line, line
    assert f"potion_swap={'off' if expect_on else 'ON'}" not in line, line
    if not expect_on:
        assert "SILENTLY DROPPED" in line, (
            "the off state must say what is LOST, not merely that a flag is "
            "unset — the v2 launcher's author read 'off' as harmless"
        )


def test_the_banner_reads_the_same_resolver_the_env_hands_to_runmanager() -> None:
    """A banner with its own source of truth can lie, which is worse than none.

    `PpoRunEnv` builds its `RunManager` with
    `interactive_potion_swap=sts2_rs.resolve_potion_swap_flag()`. If the banner
    consulted `os.environ` itself it could drift from that — different parsing,
    a different variable name after a rename — and report a world the workers
    are not building.
    """
    import sts2_rs
    from ppo.trainer import potion_swap_banner

    on = sts2_rs.resolve_potion_swap_flag()
    assert f"potion_swap={'ON' if on else 'off'}" in potion_swap_banner()


def test_a_malformed_flag_fails_at_startup_not_inside_a_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo must stop the run before it spends an hour collecting.

    `resolve_potion_swap_flag` panics on anything but unset/"0"/"1" (a
    measurement seam with no silent fallback). Calling it from the banner
    means that panic lands in the first second of the campaign instead of in a
    spawned collection worker, where it reads as a flaky seed.
    """
    from ppo.trainer import potion_swap_banner

    monkeypatch.setenv("STS2_RUNAI_POTION_SWAP", "true")
    with pytest.raises(BaseException, match="STS2_RUNAI_POTION_SWAP"):
        potion_swap_banner()
