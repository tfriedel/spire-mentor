"""Structural contracts for the workbench's faithful play via HeuristicRunAgent.

After Task 4, ``BranchingLiveSession.start_faithful`` drives the native
``HeuristicRunAgent`` (not the old ``RunStepper``/``play_run_mcts`` path).
These tests verify the structural invariants that still hold:

- A faithful run opens on the Neow event (neow_start flag is set).
- ``play_to_end`` completes the run without error.
- Looping ``faithful_step()`` to completion finishes without error and records
  one branch node per step.
- Pause/resume via ``faithful_step()`` reaches the same terminal state as an
  uninterrupted ``play_to_end``.
- ``seek()`` correctly replays agent-native action nodes and doesn't diverge.
- Regret review ``node_id`` fields point at the right branch-tree nodes and
  seek survives a rewind.
- The HTTP surface (start-faithful + play-to-end + faithful-step) works
  end-to-end with the new request model (no ``reward_sims`` required).

The old "must reproduce play_run_mcts exactly" assertion has been removed:
the workbench now drives the native HeuristicRunAgent, which is the policy
under test.  Budget defaults are eval-cheap (1000 / 4000); tests use a
reduced budget (sims=48, boss_sims=96) so they complete in a few seconds.
"""

from __future__ import annotations

import pytest

sts2_rs = pytest.importorskip("sts2_rs")

from run_policy import to_workbench_verb  # noqa: E402
from workbench import render  # noqa: E402
from workbench.branching import BranchingLiveSession  # noqa: E402

# Small budgets keep each full run fast while still exercising the agent.
# The HeuristicRunAgent is deterministic for a fixed seed at a fixed budget,
# so two runs with the same params reach the same terminal state.
SIMS, BOSS_SIMS = 48, 96
SEEDS = [1, 2, 3]


@pytest.mark.parametrize("seed", SEEDS)
def test_faithful_run_opens_on_neow(seed: int, tmp_path) -> None:
    """A faithful run must begin at the Neow event, like the real game and the
    run-AI arena (``arena.rs`` ``with_neow_start``).

    Regression: ``start_faithful`` used to construct the ``RunManager`` without
    ``neow_start``, so the run started at ``Phase::MapChoice`` — silently
    skipping the opening blessing AND shifting the act-1 event cursor. The first
    decision must be the Neow ``event_choice``, not ``map_select``.
    """
    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="rust", loc={})
    state = sess.start_faithful(
        "Ironclad",
        seed,
        sims=SIMS,
        boss_sims=BOSS_SIMS,
        threads=1,
    )
    assert state.get("decision") == "event_choice", (
        f"seed {seed}: faithful run should open on the Neow event, got {state.get('decision')!r}"
    )
    assert state.get("event_name"), f"seed {seed}: expected a Neow event name"


@pytest.mark.parametrize("seed", SEEDS)
def test_workbench_play_to_end_completes(seed: int, tmp_path) -> None:
    """play_to_end() runs a faithful seed to completion without error.

    Replaces the old "matches play_run_mcts" assertion: the workbench now
    drives the HeuristicRunAgent directly (not the RunStepper), so there is no
    external reference to compare against.  The contract is: completes, no
    error, over=True.
    """
    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="rust", loc={})
    sess.start_faithful(
        "Ironclad",
        seed,
        sims=SIMS,
        boss_sims=BOSS_SIMS,
        threads=1,
    )
    out = sess.play_to_end(max_steps=4000)

    assert out["error"] is None, f"seed {seed}: autoplay errored: {out['error']}"
    assert out["over"], f"seed {seed}: run did not finish (steps={out['steps']})"
    assert out["steps"] > 0, f"seed {seed}: expected at least one step"


def test_faithful_endpoints_work(tmp_path) -> None:
    """The HTTP surface (start-faithful + play-to-end) works with the new
    request model (no reward_sims required; sims/boss_sims use new defaults)."""
    from fastapi.testclient import TestClient
    from workbench.server import create_app

    (tmp_path / "images").mkdir()
    (tmp_path / "logs").mkdir()
    client = TestClient(create_app(images_root=tmp_path / "images", logs_dir=tmp_path / "logs"))

    r = client.post(
        "/api/live/start-faithful",
        json={
            "character": "Ironclad",
            "seed": "run_1",
            "sims": SIMS,
            "boss_sims": BOSS_SIMS,
            "threads": 1,
        },
    )
    assert r.status_code == 200
    body = client.post("/api/live/play-to-end").json()
    assert body["error"] is None
    assert body["over"]
    # One branch node per applied step, plus the root.
    assert len(body["timeline"]["nodes"]) == body["steps"] + 1


def test_faithful_endpoints_ignore_stale_reward_sims(tmp_path) -> None:
    """A stale SPA posting reward_sims must not 500 — Pydantic ignores extras."""
    from fastapi.testclient import TestClient
    from workbench.server import create_app

    (tmp_path / "images").mkdir()
    (tmp_path / "logs").mkdir()
    client = TestClient(create_app(images_root=tmp_path / "images", logs_dir=tmp_path / "logs"))

    r = client.post(
        "/api/live/start-faithful",
        json={
            "character": "Ironclad",
            "seed": "run_1",
            "sims": SIMS,
            "boss_sims": BOSS_SIMS,
            "reward_sims": 3000,  # stale SPA field — must be accepted and ignored
            "threads": 1,
        },
    )
    assert r.status_code == 200, f"unexpected status {r.status_code}: {r.text}"


def test_faithful_step_loop_completes(tmp_path) -> None:
    """Looping faithful_step() to completion records one branch node per step.

    Replaces the old "step count == play_run_mcts steps" assertion.
    """
    seed = 1

    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="rust", loc={})
    sess.start_faithful(
        "Ironclad",
        seed,
        sims=SIMS,
        boss_sims=BOSS_SIMS,
        threads=1,
    )
    out = {"done": False}
    steps = 0
    for _ in range(4000):
        out = sess.faithful_step()
        steps += 1
        if out["done"] or out["over"]:
            break

    assert out["error"] is None, f"step loop errored: {out['error']}"
    assert out["over"], "run did not finish"
    assert steps > 0
    # Root node + one node per applied action.
    assert len(sess._nodes) == steps + 1


def test_faithful_step_resume_matches_uninterrupted(tmp_path) -> None:
    """Pausing after K steps and resuming reaches the same terminal state as an
    uninterrupted run (the persisted agent is not perturbed by pausing)."""
    seed = 2

    # Uninterrupted run.
    sess_ref = BranchingLiveSession(logs_dir=str(tmp_path / "ref"), backend="rust", loc={})
    (tmp_path / "ref").mkdir()
    sess_ref.start_faithful("Ironclad", seed, sims=SIMS, boss_sims=BOSS_SIMS, threads=1)
    ref_out = sess_ref.play_to_end(max_steps=4000)

    # Paused-then-resumed run.
    (tmp_path / "resume").mkdir()
    sess_res = BranchingLiveSession(logs_dir=str(tmp_path / "resume"), backend="rust", loc={})
    sess_res.start_faithful("Ironclad", seed, sims=SIMS, boss_sims=BOSS_SIMS, threads=1)
    # "Pause" after 5 steps, then "resume" by continuing to call faithful_step().
    for _ in range(5):
        if sess_res.faithful_step()["done"]:
            break
    out = {"done": False}
    for _ in range(4000):
        out = sess_res.faithful_step()
        if out["done"] or out["over"]:
            break

    assert out["over"]
    assert out["victory"] == ref_out["victory"]
    # out["steps"] from faithful_step() is the cumulative total for the whole run
    # (5 pause steps + resume steps).  ref_out["steps"] from play_to_end() is
    # the total steps taken in that call.  Same seed+params → same total.
    assert out["steps"] == ref_out["steps"]


def test_faithful_seek_replays_without_diverging(tmp_path) -> None:
    """Seeking back into a faithful run must rebuild the engine the SAME way
    ``start_faithful`` did (integer seed, ``neow_start`` on) so the recorded
    actions replay cleanly.

    Regression: ``seek()`` used to restart every run via the interactive
    ``start()``. That path takes a TEXT seed (``string_seed``), which hashes
    to a different run than the faithful integer seed, so the recorded actions
    replayed against the wrong run and were rejected with
    ``invalid_for_phase``. (Pre-GH #399 the two constructors also differed in
    the ``interactive_card_*`` / ``direct_event_relic_grant`` flags; they no
    longer do — the seed type is what is left.)
    """
    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="rust", loc={})
    sess.start_faithful(
        "Ironclad",
        1,
        sims=SIMS,
        boss_sims=BOSS_SIMS,
        threads=1,
    )
    # Advance a handful of steps so there's a path of recorded nodes to replay.
    for _ in range(6):
        if sess.faithful_step()["done"]:
            break

    # Node 1 is the first action past the root; it triggered the original crash.
    expected_decision = sess._nodes[1].decision
    state = sess.seek(1)  # must not raise "replay diverged ... invalid_for_phase"

    assert sess._active_id == 1
    assert state.get("decision") == expected_decision


def test_faithful_regret_click_seek_into_combat(tmp_path) -> None:
    """The post-combat regret view's "click a decision to jump there" path: a
    blunder's ``idx`` resolves to the combat node on the active line, and seeking
    there must replay cleanly into mid-combat.

    Regression for the user-visible "clicking decisions doesn't jump": seek into
    a faithful run replayed through the interactive ``start()`` and was rejected
    with ``invalid_for_phase`` (the 500 the UI's ``seekTo`` swallowed silently),
    so combat seek targets — not just node 1 — must round-trip.
    """
    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="rust", loc={})
    sess.start_faithful(
        "Ironclad",
        1,
        sims=SIMS,
        boss_sims=BOSS_SIMS,
        threads=1,
    )
    # Play far enough to fight at least one combat (several card-play nodes).
    for _ in range(60):
        if sess.faithful_step()["done"]:
            break

    review = sess.review_run({})
    # Mirror the frontend's liveSeekTargetForIdx: map each reviewed decision's
    # depth to the active-path node at that depth, then seek there.
    on_path = set(sess.timeline()["active_path"])
    by_idx = {n.idx: n.id for n in sess._nodes.values() if n.id in on_path}
    combat_targets = [by_idx[d["idx"]] for d in review["decisions"] if d["idx"] in by_idx]
    assert combat_targets, "expected at least one combat decision to seek to"

    for node_id in combat_targets:
        state = sess.seek(node_id)  # must not raise invalid_for_phase
        assert sess._active_id == node_id
        assert state.get("decision") == "combat_play"


def test_regret_decisions_carry_node_id_and_seek_survives_rewind(tmp_path) -> None:
    """Each regret decision carries the branch-tree node it was chosen from, and
    seeking to a *later* decision still works after an earlier seek has rewound
    (truncated) the active path.

    Regression for "clicking decisions stops jumping after the first rewind": the
    frontend used to map a decision to a node by depth-on-active-path, so once a
    rewind shortened the active line every later decision resolved to nothing.
    The node id is stable, and ``seek`` replays any node from the root regardless
    of the current active leaf.
    """
    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="rust", loc={})
    sess.start_faithful(
        "Ironclad",
        1,
        sims=SIMS,
        boss_sims=BOSS_SIMS,
        threads=1,
    )
    for _ in range(60):
        if sess.faithful_step()["done"]:
            break

    decisions = sess.review_run({})["decisions"]
    assert decisions, "expected combat decisions to review"
    # Every decision points at the combat node it was chosen from.
    for d in decisions:
        node = sess._nodes[d["node_id"]]
        assert node.idx == d["idx"]
        assert node.state.get("decision") == "combat_play"

    if len(decisions) >= 2:
        early, late = decisions[0]["node_id"], decisions[-1]["node_id"]
        # Rewind to the EARLY decision: this truncates the active path so the
        # late decision's node is now on a pruned sibling branch.
        sess.seek(early)
        assert late not in set(sess.timeline()["active_path"])
        # Seeking to the late decision by node id must still land there.
        state = sess.seek(late)
        assert sess._active_id == late
        assert state.get("decision") == "combat_play"


def test_faithful_step_endpoint_loops_to_completion(tmp_path) -> None:
    """Looping POST /api/live/faithful-step completes the run and records one
    branch node per step (the live UI's transport)."""
    from fastapi.testclient import TestClient
    from workbench.server import create_app

    (tmp_path / "images").mkdir()
    (tmp_path / "logs").mkdir()
    client = TestClient(create_app(images_root=tmp_path / "images", logs_dir=tmp_path / "logs"))

    r = client.post(
        "/api/live/start-faithful",
        json={
            "character": "Ironclad",
            "seed": "run_1",
            "sims": SIMS,
            "boss_sims": BOSS_SIMS,
            "threads": 1,
        },
    )
    assert r.status_code == 200

    loop_count = 0
    body = {"done": False}
    for _ in range(4000):
        body = client.post("/api/live/faithful-step").json()
        loop_count += 1
        if body["done"]:
            break

    assert body["error"] is None
    assert body["over"] is True
    # Terminal call returns done with no new action applied, so applied
    # steps == loop_count - 1.
    assert loop_count > 1, "expected more than one loop iteration"
    assert loop_count - 1 == body["steps"], (
        f"expected applied steps == loop_count-1, got steps={body['steps']} loop_count={loop_count}"
    )


def test_faithful_action_label_play_card_renders_name() -> None:
    """play_card nodes must render "Play <CardName>" not "Play None".

    Regression: faithful_step() passed the native RunAction dict (which uses
    ``hand_idx``) directly to ``render.describe_action`` (which expects
    ``card_index``).  The fix translates via ``to_workbench_verb`` first.

    This is a fast unit test — no sts2_rs engine needed.
    """
    parent_state = {
        "hand": [
            {"id": "Strike_R", "name": "Strike", "upgraded": False, "cost": 1},
            {"id": "Defend_R", "name": "Defend", "upgraded": False, "cost": 1},
        ],
        "enemies": [{"id": "Cultist", "name": "Cultist", "current_hp": 50, "max_hp": 50}],
    }
    # Native Rust play_card action: hand_idx (not card_index).
    native_action = {"action": "play_card", "hand_idx": 0, "target_index": 0}
    verb, wb_args = to_workbench_verb(None, native_action)
    label = render.describe_action(parent_state, {"cmd": "action", "action": verb, "args": wb_args})
    assert "Strike" in label, f"expected card name in label, got {label!r}"
    assert "None" not in label, f"label must not contain 'None', got {label!r}"

    # Also verify end_turn and a second card index.
    end_verb, end_args = to_workbench_verb(None, {"action": "end_turn"})
    end_label = render.describe_action(
        parent_state, {"cmd": "action", "action": end_verb, "args": end_args}
    )
    assert end_label == "End turn"

    card1_verb, card1_args = to_workbench_verb(None, {"action": "play_card", "hand_idx": 1})
    card1_label = render.describe_action(
        parent_state, {"cmd": "action", "action": card1_verb, "args": card1_args}
    )
    assert "Defend" in card1_label, f"expected 'Defend' in label, got {card1_label!r}"


def test_faithful_save_line_reloads_without_diverging(tmp_path) -> None:
    """A named save of a run-mode ("play to end") run must load back.

    Regression for the deployed workbench's ``409 replay diverged at action
    'select_card_reward': ... invalid_for_phase``. ``load_line`` restarted
    EVERY save through the interactive ``start()``, which takes a TEXT seed
    (``string_seed``) and hashes to a different run than the run-mode integer
    seed — so the saved line replayed against an unrelated game and the engine
    rejected the first action whose phase no longer matched, several steps in.
    ``seek()`` has always re-derived the constructor; saves had to record it.
    """
    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="rust", loc={})
    sess.start_faithful("Ironclad", 1, sims=SIMS, boss_sims=BOSS_SIMS, threads=1)
    for _ in range(40):
        if sess.faithful_step()["done"]:
            break
    assert sess.run_mode is True
    line, _progress = sess.active_line()
    assert any(e["action"] == "__agent_native__" for e in line), "expected agent-native nodes"
    leaf = sess.state()

    restored = BranchingLiveSession(logs_dir=str(tmp_path), backend="rust", loc={})
    state = restored.load_line("Ironclad", sess.seed, 0, line, run_mode=True)
    assert state.get("decision") == leaf.get("decision")
    assert state.get("player", {}).get("hp") == leaf.get("player", {}).get("hp")
    # The rebuilt tree carries the whole line, so the user can still rewind.
    assert len(restored.active_line()[0]) == len(line)


def test_loading_a_run_mode_save_as_interactive_diverges(tmp_path) -> None:
    """The pre-fix behavior, pinned: restoring with the wrong constructor fails.

    This is what made the bug invisible — the engine accepts the early
    index-shaped actions in the *other* run too, so the rejection lands
    somewhere in the middle of the line rather than at step 1.
    """
    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="rust", loc={})
    sess.start_faithful("Ironclad", 1, sims=SIMS, boss_sims=BOSS_SIMS, threads=1)
    for _ in range(40):
        if sess.faithful_step()["done"]:
            break
    line, _ = sess.active_line()

    restored = BranchingLiveSession(logs_dir=str(tmp_path), backend="rust", loc={})
    with pytest.raises(RuntimeError, match="replay diverged at step"):
        restored.load_line("Ironclad", sess.seed, 0, line, run_mode=False)


def test_quit_writes_a_summary_row_so_the_run_is_listed(tmp_path) -> None:
    """A played run must appear on the "MCTS Runs" page.

    ``workbench.aggregate._parse_run`` drops any log without a terminal
    ``summary`` meta row, and ``quit()`` never wrote one — so hand-played and
    autoplayed runs were invisible there ("runs 0") even though their JSONL sat
    in the same logs dir as the recorder's.
    """
    from pathlib import Path

    from workbench.aggregate import _parse_run

    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="rust", loc={})
    sess.start_faithful("Ironclad", 1, sims=SIMS, boss_sims=BOSS_SIMS, threads=1)
    for _ in range(20):
        if sess.faithful_step()["done"]:
            break
    out = sess.quit()

    parsed = _parse_run(Path(out["log_path"]))
    assert parsed is not None, "the run's own log must parse as a run"
    assert parsed["final_floor"] is not None
    assert parsed["won"] is False  # a 20-step run has not won


def test_discard_leaves_no_log(tmp_path) -> None:
    """A diverged restore must not litter the logs dir with a partial run."""
    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="rust", loc={})
    sess.start_faithful("Ironclad", 1, sims=SIMS, boss_sims=BOSS_SIMS, threads=1)
    sess.faithful_step()
    before = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    sess.discard()
    assert sorted(p.name for p in tmp_path.glob("*.jsonl")) == before


def test_save_load_endpoints_round_trip_a_run_mode_run(tmp_path) -> None:
    """End-to-end reproduction of the deployed workbench's unloadable save.

    Recorded flow (Cloud Run, 2026-08-24): ``POST /api/live/start-faithful`` →
    autoplay → ``POST /api/live/save`` → ``POST /api/saves/load`` → ``409
    replay diverged at action 'select_card_reward': ... invalid_for_phase``.
    """
    from fastapi.testclient import TestClient
    from workbench.server import create_app

    (tmp_path / "images").mkdir()
    (tmp_path / "logs").mkdir()
    client = TestClient(create_app(images_root=tmp_path / "images", logs_dir=tmp_path / "logs"))

    client.post(
        "/api/live/start-faithful",
        json={"character": "Ironclad", "seed": "run_1", "sims": SIMS, "boss_sims": BOSS_SIMS},
    )
    for _ in range(30):
        if client.post("/api/live/run-mode-step").json()["done"]:
            break
    before = client.get("/api/live/state").json()
    assert client.post("/api/live/save", json={"name": "ironclad A10"}).status_code == 200
    client.post("/api/live/quit")

    r = client.post("/api/saves/load", json={"name": "ironclad A10"})
    assert r.status_code == 200, r.json()
    assert r.json()["decision"] == before["decision"]


def test_loading_a_legacy_save_without_run_mode_still_works(tmp_path) -> None:
    """Saves written before the ``run_mode`` field must still load.

    The loader probes both constructors for them — the numeric seed says
    run-mode first — so the saves already sitting on the deployed instance are
    not stranded by the fix.
    """
    import json

    from fastapi.testclient import TestClient
    from workbench import saves as saves_mod
    from workbench.server import create_app

    (tmp_path / "images").mkdir()
    (tmp_path / "logs").mkdir()
    client = TestClient(create_app(images_root=tmp_path / "images", logs_dir=tmp_path / "logs"))

    client.post(
        "/api/live/start-faithful",
        json={"character": "Ironclad", "seed": "run_1", "sims": SIMS, "boss_sims": BOSS_SIMS},
    )
    for _ in range(30):
        if client.post("/api/live/run-mode-step").json()["done"]:
            break
    before = client.get("/api/live/state").json()
    client.post("/api/live/save", json={"name": "legacy"})
    client.post("/api/live/quit")

    # Age the file back to the pre-fix schema.
    path = saves_mod.save_dir(tmp_path / "logs") / "legacy.json"
    payload = json.loads(path.read_text())
    del payload["run_mode"]
    path.write_text(json.dumps(payload))

    r = client.post("/api/saves/load", json={"name": "legacy"})
    assert r.status_code == 200, r.json()
    assert r.json()["decision"] == before["decision"]


class _GrantGrid:
    """The slice of ``RunManager`` a card-grant grid shows the translator."""

    def __init__(self, options: int, minimum: int) -> None:
        self.pending_card_grant_min = minimum
        self._options = [("Strike", False)] * options

    def card_grant_options(self) -> list[tuple[str, bool]]:
        return self._options


def test_declining_an_optional_card_grant_is_a_skip_not_a_card() -> None:
    """Sea Glass offers "up to N" cards; the engine spells "none" as a grant at
    ``idx == option count``. That is a skip, not ``select_cards 15``: the
    backend reads an index as a card and appends its own confirm after it, a
    second action the engine rejects (the PPO brain's run stopped there).
    """
    grid = _GrantGrid(options=15, minimum=0)
    decline = {"action": "grant_deck_card", "idx": 15}
    assert to_workbench_verb(grid, decline) == ("skip_select", {})
    # A real pick on the same grid is still a pick.
    pick = {"action": "grant_deck_card", "idx": 14}
    assert to_workbench_verb(grid, pick) == ("select_cards", {"indices": "14"})


def test_a_mandatory_card_grant_has_no_decline_index() -> None:
    """A fixed-count grant (Room Full of Cheese) has no "none": every index
    the engine names is a card."""
    grid = _GrantGrid(options=3, minimum=3)
    assert to_workbench_verb(grid, {"action": "grant_deck_card", "idx": 2}) == (
        "select_cards",
        {"indices": "2"},
    )
    # Other pending selects never read the grant grid.
    assert to_workbench_verb(grid, {"action": "upgrade_deck_card", "idx": 3}) == (
        "select_cards",
        {"indices": "3"},
    )
