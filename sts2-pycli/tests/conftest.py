"""Pytest fixtures: Game process wrapper for unit tests.

The ``Game`` class itself lives in ``python/sts2_client.py``; this module just
binds it as a pytest fixture. ``python/`` is added to ``sys.path`` via the
``pythonpath`` setting in ``pyproject.toml``.

Two kinds of inputs are absent from a bring-your-own-game checkout (the public
export) until ``just import-game`` has run, and the suite must degrade to
SKIPS — never to failures — without them:

* The C# oracle links against the player's own game assembly
  (``lib/sts2.dll``; never published). Tests that need it carry the ``oracle``
  marker — applied here to every user of the ``game`` fixture — and are
  skipped when the assembly is absent, so ``pytest -m "not oracle"`` is the
  game-free subset and a plain ``pytest`` degrades to skips instead of ``EOF
  from game process`` noise. Tests that build ``Game()`` themselves are skipped
  the same way when the constructor raises ``OracleUnavailable``.
* Game-derived data files (``localization_eng/``, the sibling wiki's
  ``data/<ver>/``, ``workbench/data/card_db.json``), training checkpoints
  (``data/ppo/*.pt``) and private deploy artifacts (root ``Dockerfile``,
  ``web/wrangler.toml``). ``_NEEDS`` below maps the test modules / tests that
  read them to the file that must exist; they get the ``game_data`` marker and
  skip when it is missing.
"""

import os
from pathlib import Path

import pytest
import realgame_instinct
import sts2_client
from sts2_client import Game, OracleUnavailable, oracle_available

ORACLE_SKIP_REASON = "C# oracle unavailable: lib/sts2.dll missing (run `just import-game`)"


def pytest_addoption(parser):
    parser.addoption(
        "--game-version",
        action="store",
        default=None,
        help=(
            "game build the C# oracle tests run against "
            f"({', '.join(sts2_client.SUPPORTED_GAME_VERSIONS)}; default: "
            f"${sts2_client.GAME_VERSION_ENV} or {sts2_client.DEFAULT_GAME_VERSION}). "
            "Oracle tests skip unless lib/<version>/ is staged and "
            "`just build-cs <version>` has run."
        ),
    )


#: Math-library thread pools an xdist worker gets. Each of the `-n auto`
#: workers otherwise sizes torch's and BLAS's pools to every core, so 48
#: workers ran ~48 threads each and the torch-heavy tests spent their time
#: contending (2026-11-16, v0.111.0 suite on 48 cores: 85 -> 60 CPU-minutes;
#: `turnplan/test_train.py::test_learns_synthetic_best_feature` 93 s -> 1 s).
#: Set before any test module imports torch; an explicit setting wins.
_WORKER_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def pytest_configure(config):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        for var in _WORKER_THREAD_VARS:
            os.environ.setdefault(var, "1")
    requested = config.getoption("--game-version")
    if requested:
        try:
            sts2_client.set_game_version(requested)
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from exc
    # Children (xdist workers, the headless oracle, parity pool workers)
    # inherit the selection through the environment.
    os.environ[sts2_client.GAME_VERSION_ENV] = sts2_client.GAME_VERSION


def _oracle_skip_reason() -> str | None:
    """Why the selected game build's C# oracle cannot run, or None if it can."""
    ver = sts2_client.GAME_VERSION
    if not oracle_available():
        return f"C# oracle unavailable: lib/{ver}/sts2.dll missing (run `just import-game`)"
    if not sts2_client.oracle_built():
        return f"C# oracle for {ver} not built (run `just build-cs {ver}`)"
    return None


def pytest_report_header(config):
    reason = _oracle_skip_reason()
    return f"game version: {sts2_client.GAME_VERSION} (C# oracle: {reason or 'ready'})"


_PYCLI = Path(__file__).resolve().parents[1]
_MONO = _PYCLI.parent

# Requirement name -> (path that must exist, how to obtain it).
_REQUIREMENTS: dict[str, tuple[Path, str]] = {
    "localization": (_PYCLI / "localization_eng" / "cards.json", "just import-game --tier oracle"),
    "wiki_data": (_MONO / "sts2-wiki" / "data" / "v0.107.1" / "cards.json", "just import-game"),
    # The newer builds' wiki data stays private (unit-mono.toml ships v0.107.1's).
    "wiki_data_v0111": (
        _MONO / "sts2-wiki" / "data" / "v0.111.0" / "cards.json",
        "just import-game",
    ),
    "wiki_data_v0113": (
        _MONO / "sts2-wiki" / "data" / "v0.113.0" / "cards.json",
        "just import-game",
    ),
    "card_db": (_PYCLI / "python" / "workbench" / "data" / "card_db.json", "just import-game"),
    "checkpoints": (
        _PYCLI / "data" / "ppo" / "ppo_fullrun_it350.pt",
        "private training checkpoints (.pt)",
    ),
    "dockerfile": (_MONO / "Dockerfile", "private deploy artifact"),
    "wrangler": (_PYCLI / "web" / "wrangler.toml", "private deploy artifact"),
    "combat_bench": (
        _PYCLI / "data" / "combat_bench" / "ironclad_dev_v1.jsonl",
        "private: built from the scraped human-run corpus (D6)",
    ),
    "rollout_recipe": (
        _MONO / "logs" / "2026-10-18-rollout-net-recipe" / "boss800.jsonl",
        "private run logs (logs/ is not exported)",
    ),
    "instinct_corpus": (
        _MONO / "logs" / "2026-11-07-instinct-live" / "run1-requests.jsonl.gz",
        "private run logs (logs/ is not exported) + the private Instinct net (data/e2e)",
    ),
    "host_overrides": (
        _PYCLI / "data" / "host_overrides.json",
        "names game methods; not exported (GH #1299)",
    ),
    "decomp_v0111": (
        _MONO / "sts2-wiki" / "decompiled" / "v0.111.0",
        "the C# decompilation (decompiled/ is not exported)",
    ),
    "decomp_all": (
        _MONO / "sts2-wiki" / "decompiled" / "v0.113.1",
        "the C# decompilation of every supported build (decompiled/ is not exported)",
    ),
    # The net the advisors load, so a swap of the net cannot turn this into a
    # silent skip on the private repo.
    "instinct_net": (
        _PYCLI / realgame_instinct.WEIGHTS_RELPATH,
        "the private Instinct net (data/e2e is not exported)",
    ),
}

# Node-id prefix (relative to tests/) -> requirement. A bare module prefix
# covers every test in the module; ``module::test`` covers one test.
_NEEDS: dict[str, str] = {
    # --- the vendored English localization
    "test_localization_registry_coverage.py": "localization",
    "test_realgame_advisor_deck_mutation.py": "localization",
    "test_display_names.py::test_names_come_from_the_game_tables": "localization",
    "test_rust_game_backend.py::test_csharp_resolve_event_fields_resolves_boss_relic_and_regular_event": "localization",
    "test_hint_adapter.py::test_ironclad_vs_nibbit_mid_turn": "localization",
    "test_hint_adapter.py::test_mid_turn_with_potions": "localization",
    "test_hint_adapter.py::test_the_title_path_still_wins_when_the_id_is_not_a_registry_name": "localization",
    "test_hint_adapter.py::test_wax_and_melt_flags_survive_translation": "localization",
    "workbench/test_regret_review.py": "localization",
    "workbench/test_bridge_tree_log.py": "localization",
    "workbench/test_branching.py": "localization",
    "workbench/test_timeline_details.py": "localization",
    "workbench/test_localization.py": "localization",
    "workbench/test_chains_of_binding.py": "localization",
    "workbench/test_card_text_placeholders.py": "localization",
    "workbench/test_event_option_loc_keys.py": "localization",
    "workbench/test_monster_map_sync.py": "localization",
    "workbench/test_orb_map_sync.py": "localization",
    "workbench/test_server.py::test_winged_boots_loc_key_resolves_from_relics_json": "localization",
    # --- the sibling wiki's extracted data (sts2-lookup, boss move titles)
    "test_play.py::test_boss_known_moves_includes_localized_titles": "wiki_data",
    "test_play.py::test_show_boss_renders_name_and_moves": "wiki_data",
    "test_sts2_mcp.py::TestRecommendCombat::test_recommend_picks_a_card": "wiki_data",
    "test_mcp_recommend_combat_mcts.py::TestRecommendCombatMctsHappyPath": "wiki_data",
    "test_mcp_recommend_combat_mcts.py::TestRecommendCombatUntouched": "wiki_data",
    "test_advisor_mirror.py::test_a_potion_screen_is_searched_end_to_end": "wiki_data",
    # --- a newer build's glossary: the generator reads that build's wiki data
    # and the shipped file sits in web/public/<ver>/, both private (GH #1615)
    "workbench/test_gen_glossary.py::test_shipped_glossary_is_fresh[v0.111.0]": "wiki_data_v0111",
    "workbench/test_gen_glossary.py::test_shipped_glossary_is_fresh[v0.113.0]": "wiki_data_v0113",
    "workbench/test_gen_glossary.py::test_glossary_names_what_the_page_reads": "wiki_data_v0113",
    "workbench/test_gen_glossary.py::test_shipped_zh_glossary_is_fresh[v0.111.0]": "wiki_data_v0111",
    "workbench/test_gen_glossary.py::test_shipped_zh_glossary_is_fresh[v0.113.0]": "wiki_data_v0113",
    "workbench/test_gen_glossary.py::test_zh_glossary_is_the_english_one_in_the_games_chinese": "wiki_data_v0113",
    # --- the extracted card database the deck analyzer reads
    "workbench/test_deck_analysis.py": "card_db",
    # --- torch checkpoints (only the champion .onnx files are published)
    "test_ppo_onnx_policy.py": "checkpoints",
    "test_ppo_features_from_checkpoint.py": "checkpoints",
    "test_champion_resolvers_agree.py": "checkpoints",
    "test_ppo_shared_builders.py::test_ppo_policy_evaluate_and_recommend": "checkpoints",
    "test_ppo_shared_builders.py::test_ppo_policy_evaluate_degrades_on_inference_error": "checkpoints",
    "test_ppo_shared_builders.py::test_session_run_policy_selector": "checkpoints",
    "test_ppo_workbench_integration.py::test_snapshot_greedy_full_run_is_legal": "checkpoints",
    # --- private deploy artifacts
    "test_deploy_art_assets.py": "dockerfile",
    "test_deployed_image_has_lazy_import_deps.py": "dockerfile",
    "workbench/test_dockerfile_ships_ppo_onnx.py": "dockerfile",
    "test_bug_reports_cli.py": "wrangler",
    # --- the frozen combat-bench fight sets (human-corpus-derived, not exported)
    "turnplan/test_binding.py::test_labels_on_first_decision_of_a_dev_fight": "combat_bench",
    "turnplan/test_binding.py::test_labels_do_not_perturb_the_agents_play": "combat_bench",
    "turnplan/test_label.py::test_excluded_keys_never_labelled": "combat_bench",
    "turnplan/test_label.py::test_one_label_per_turn_start": "combat_bench",
    "turnplan/test_label.py::test_hallway_p_zero_labels_no_hallway_turn": "combat_bench",
    "turnplan/test_label.py::test_resume_skips_done_cells": "combat_bench",
    "turnplan/test_label.py::test_resume_refuses_a_changed_config": "combat_bench",
    "turnplan/test_label.py::test_a_cell_killed_mid_run_is_replayed_and_read_once": "combat_bench",
    "turnplan/test_label.py::test_a_cell_over_the_timeout_is_booked_as_watchdog": "combat_bench",
    "turnplan/test_pro.py::test_pro_set_is_the_frozen_bench_file": "combat_bench",
    "turnplan/test_pro.py::test_turn_plan_labels_locates_each_enumerated_plan_unpriced": "combat_bench",
    "turnplan/test_pro.py::test_turn_plan_labels_reports_an_unplayable_turn": "combat_bench",
    "turnplan/test_pro.py::test_leak_guard_panels_are_dev_devpool_holdout_and_pro": "combat_bench",
    "turnplan/test_root_filter.py::test_a_cell_without_the_knob_has_no_root_filter_field": "combat_bench",
    "turnplan/test_root_filter.py::test_a_cell_with_the_knob_reports_a_filtered_root": "combat_bench",
    # --- the rollout-net recipe's boss800 holdout list (private logs/)
    "test_combat_ctx.py::test_jobs_hold_out_every_holdout_run": "rollout_recipe",
    "test_realgame_instinct_live.py": "instinct_corpus",
    "test_realgame_advisor_replay.py": "instinct_corpus",
    "test_board_import_parity.py::test_every_live_board_gets_the_same_wire_from_rust_and_python": "instinct_corpus",
    # --- the combat-PPO start pool subtracts every combat-bench run
    "test_combat_ppo.py::test_train_pool_holds_out_every_holdout_run": "combat_bench",
    "test_elite_gap_tools.py::test_takeover_forces_a_human_line_exactly": "combat_bench",
    "test_elite_gap_tools.py::test_takeover_records_both_first_turns_with_rollouts": "combat_bench",
    "test_leaf_correction.py::test_play_and_label_rebuilds_the_final_value": "combat_bench",
    # --- the e2e F9 ledger's card types, checked against the v0.111.0 decomp
    # --- the reviewed override table (and the decomp the IlPatcher rows hash)
    "test_host_override_fingerprints.py": "host_overrides",
    # --- the content universe and the id lists, checked against every build's decomp
    "test_content_universe.py": "decomp_all",
    "e2e/test_e2e_invariants.py::test_no_v0111_power_is_typed_as_a_non_power": "decomp_v0111",
    "e2e/test_e2e_invariants.py::test_the_instanced_powers_match_the_decomp": "decomp_v0111",
    # --- the shipped Instinct net (GH #1531)
    "test_stage_advisor.py::test_the_staged_instinct_net_is_the_one_both_advisors_load": "instinct_net",
}


#: Wall-clock-timing tests that flake on the public repo's GitHub Actions
#: runners (2 slow cores). Node-id prefixes as in `_NEEDS`, parameters ignored.
#: They skip when `GITHUB_ACTIONS` is set: this repo has no Actions CI, so
#: that means the public CI. They still run on every local gate.
_FLAKY_ON_GITHUB_ACTIONS: dict[str, str] = {
    # state.pt lagged the stats rows by 12 iterations at the kill (138 < 149).
    "e2e/test_e2e_t11_sigkill.py::test_sigkill_then_resume_reproduces_the_next_three_iterations": (
        "state.pt falls behind the stats rows on a slow runner"
    ),
    # The 0.3 s stall window fired in the healthy phase (resumed at 720, not 900).
    "e2e/test_e2e_ops_watchdog.py::test_stall_restarts_the_learner": (
        "the stall watchdog's sub-second window fires early on a slow runner"
    ),
}


#: Tests that run a training loop (an optimizer step, a learner or trainer
#: iteration, or the training lane's supervisor). They get the `training`
#: marker, which the default run and the per-PR gate deselect; `just
#: test-training` and `just gate-full` run them. Keys are node-id prefixes as
#: in `_NEEDS`; a key with `[...]` names one parametrization.
_TRAINING: frozenset[str] = frozenset(
    {
        "e2e/test_learner.py",
        "e2e/test_e2e_ops_watchdog.py",
        "turnplan/test_train.py",
        "test_combat_ppo.py",
        "test_ppo_plasticity.py::test_a_real_iteration_writes_the_diagnostic_to_stats_jsonl",
        "test_ppo_wheel_guard.py::test_a_training_run_records_what_its_workers_ran",
        "test_ppo_trainer_units.py::test_bf16_update_trains_in_fp32_weights_and_leaves_fp32_bit_identical",
        "e2e/test_e2e_mix.py::test_mix_share_resume_invariant",
    }
)

#: Tests the per-PR gate leaves to `just test-slow` (and `gate-full`) because
#: of their wall time: each took 20-60 s on 2026-11-16 (v0.111.0, 48 cores,
#: one thread per worker) and set the suite's tail. Where a test is
#: parametrized, a cheaper parametrization stays in the fast suite. They get
#: the `slow` marker, exactly as if decorated. Keys as in `_TRAINING`.
_SLOW_BY_WALL_TIME: frozenset[str] = frozenset(
    {
        "test_ppo_env_event_surface.py::test_event_rooms_reach_the_policy_with_an_identity[seed1000]",
        "test_ppo_env_event_surface.py::test_event_rooms_reach_the_policy_with_an_identity[seed1003]",
        "test_ppo_env_event_surface.py::test_event_rooms_reach_the_policy_with_an_identity[seed1005]",
        "test_ppo_env_event_surface.py::test_ordinary_events_reach_the_policy",
        "workbench/test_run_mode.py::test_run_mode_step_finishes_a_run[ppo]",
        "test_ppo_card_reward_upgrade.py::test_the_training_obs_ships_flags_aligned_with_the_ids[seed1000]",
        "e2e/test_e2e_devboss.py::test_capture_play_read_on_the_engine",
        "test_workbench_faithful_parity.py::test_faithful_step_endpoint_loops_to_completion",
        "test_workbench_faithful_parity.py::test_faithful_step_loop_completes",
        "test_workbench_faithful_parity.py::test_workbench_play_to_end_completes[1]",
        "test_realgame_advisor_replay.py::test_a_replay_is_reproducible",
        "test_realgame_advisor_replay.py::test_every_recorded_request_gets_one_reply[search]",
        "test_event_coverage.py::TestSubsetIsClean::test_no_world_failure_on_any_option[TRIAL]",
        "test_event_coverage.py::TestSubsetIsClean::test_known_residuals_are_still_real",
        "workbench/test_workbench_blitz_saves.py",
        "test_bc_human_rebuild_roster.py::test_replay_rebuild_rolls_decimillipede_openers",
        "e2e/test_e2e_f7check.py::test_the_packed_serving_path_passes_and_sees_a_leak",
        "e2e/test_e2e_f7check.py::test_the_launch_student_passes",
        "test_referee_rng.py::test_the_same_key_and_actions_reproduce_the_game_and_another_key_does_not",
        "e2e/test_e2e_chars.py::test_lockstep_all_characters[Necrobinder]",
        "e2e/test_e2e_recorder.py::test_recording_is_deterministic",
    }
)


def _listed(nodeid: str, keys: frozenset[str]) -> bool:
    """Whether a node-id prefix in `keys` covers `nodeid` (with or without its parameters)."""
    rel = nodeid.split("tests/", 1)[-1]
    if rel in keys:
        return True
    parts = rel.split("[", 1)[0].split("::")
    return any("::".join(parts[:n]) in keys for n in range(len(parts), 0, -1))


def _flaky_on_github_actions(nodeid: str) -> str | None:
    rel = nodeid.split("tests/", 1)[-1].split("[", 1)[0]
    parts = rel.split("::")
    for n in range(len(parts), 0, -1):
        hit = _FLAKY_ON_GITHUB_ACTIONS.get("::".join(parts[:n]))
        if hit:
            return hit
    return None


def _needs(nodeid: str) -> str | None:
    rel = nodeid.split("tests/", 1)[-1]
    parts = rel.split("::")
    for n in range(len(parts), 0, -1):
        hit = _NEEDS.get("::".join(parts[:n]))
        if hit:
            return hit
    return None


#: What the engine has not ported yet for a supported game build, by marker.
#: GH #1303 tracks the v0.113.0 port. A test carrying ``pending_<name>`` is a
#: STRICT xfail on a build listed here: it must fail there, and the first run in
#: which it passes goes red (XPASS), which is the cue to drop the build. The
#: values are (builds, reason, the only exception a pass may fail with or None).
#:
#: * ``run_level`` -- the engine still rolls the previous build's run level on
#:   these builds (Neow offer, act-1 boss), so a C#-captured golden cannot
#:   match yet.
#: * ``e2e_tables`` -- the e2e brain's static tables are not generated for
#:   these builds (``tests/e2e/test_e2e_static_tables.py`` ``PENDING_TABLES``).
PENDING_BUILDS: dict[str, tuple[frozenset[str], str, type[BaseException] | None]] = {
    "run_level": (
        frozenset(),
        "run level (Neow offer, act rolls) not ported to this build yet (GH #1303)",
        None,
    ),
    "e2e_tables": (
        frozenset(),
        "e2e static tables not generated for this build yet (GH #1924)",
        KeyError,
    ),
}


@pytest.fixture
def run_level_pending() -> bool:
    """Whether the active build's run level is still pending (`PENDING_BUILDS`)."""
    return sts2_client.GAME_VERSION in PENDING_BUILDS["run_level"][0]


@pytest.fixture
def pending_builds() -> dict[str, frozenset[str]]:
    """`PENDING_BUILDS` as {name: builds}."""
    return {name: builds for name, (builds, _, _) in PENDING_BUILDS.items()}


def _mark_pending(item: pytest.Item) -> None:
    for name, (builds, reason, raises) in PENDING_BUILDS.items():
        if item.get_closest_marker(f"pending_{name}") and sts2_client.GAME_VERSION in builds:
            item.add_marker(pytest.mark.xfail(strict=True, reason=reason, raises=raises))


def pytest_collection_modifyitems(items):
    oracle_skip = _oracle_skip_reason()
    for item in items:
        _mark_pending(item)
        if _listed(item.nodeid, _TRAINING):
            item.add_marker(pytest.mark.training)
        if _listed(item.nodeid, _SLOW_BY_WALL_TIME):
            item.add_marker(pytest.mark.slow)
        if "game" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.oracle)
        if oracle_skip and item.get_closest_marker("oracle"):
            item.add_marker(pytest.mark.skip(reason=oracle_skip))
        req = _needs(item.nodeid)
        if req:
            item.add_marker(pytest.mark.game_data)
            path, how = _REQUIREMENTS[req]
            if not path.exists():
                item.add_marker(pytest.mark.skip(reason=f"needs {path.relative_to(_MONO)} ({how})"))
        if os.environ.get("GITHUB_ACTIONS") == "true":
            flaky = _flaky_on_github_actions(item.nodeid)
            if flaky:
                item.add_marker(pytest.mark.skip(reason=f"timing flake on GitHub Actions: {flaky}"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    # Module/class-scoped fixtures that build their own ``Game()``.
    try:
        return (yield)
    except OracleUnavailable:
        pytest.skip(ORACLE_SKIP_REASON)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except OracleUnavailable:
        pytest.skip(ORACLE_SKIP_REASON)


@pytest.fixture(autouse=True)
def _reset_engine_unlock_state():
    """Undo any epoch lock a test installed into the engine. GH #755.

    ``sts2_rs.set_locked_epochs`` mutates PROCESS-GLOBAL engine state: every
    card pool built afterwards in the process is pruned. The live drivers'
    ``main()`` installs the live census on purpose (``realgame_mcts`` /
    ``realgame_random`` → ``install_live_unlock_state``), so a test that runs one
    leaked the lock into whatever ran next in its xdist worker — e.g. the
    Scroll Boxes golden then rolled DEATHBRINGER instead of DEATH_MARCH. The
    adapter's once-per-process wire latch is reset with it.

    Reads ``sys.modules`` instead of importing, so tests that never load the
    engine pay nothing.
    """
    yield
    import sys

    rs = sys.modules.get("sts2_rs")
    if rs is not None and hasattr(rs, "locked_cards") and rs.locked_cards():
        rs.set_locked_epochs([])
    adapter = sys.modules.get("realgame_adapter")
    if adapter is not None:
        adapter._WIRE_UNLOCKS_INSTALLED = None


@pytest.fixture
def game():
    """Each test gets an independent game process."""
    reason = _oracle_skip_reason()
    if reason:
        pytest.skip(reason)
    g = Game()
    yield g
    g.close()
