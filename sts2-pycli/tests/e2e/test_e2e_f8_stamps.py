"""F8 stamps everywhere the plan names them (plan section 3.1, GH #1097).

"Stamps: source digest, wheel, game version, vocab and schema sha256 in cfg,
checkpoints, stats and eval rows." Each writer's own tests check presence or a
subset; these check the VALUES of every `stamps.STAMP_KEYS` key against the
writing process: cfg.json, every stats row (in memory and on disk), the state,
full, pinned and slim checkpoints (top-level stamps and the net's cfg stamps), an
exported student, and the eval rows (a student's played row and an error row).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import sts2_client
import test_learner as TL  # make_cfg, make_learner, ITER, TINY: the fake source and teachers
import torch
from ppo.e2e import ckpt
from ppo.e2e import contracts as C
from ppo.e2e import learner as L
from ppo.e2e import ops_contract as OC
from ppo.e2e import stamps as ST
from ppo.e2e.net import load_checkpoint
from ppo.e2e.update import export_variant
from ppo.wheel_guard import python_source_digest, wheel_identity

V111 = "v0.111.0"

try:
    import sts2_rs  # noqa: F401

    HAVE_ENGINE = True
except ImportError:
    HAVE_ENGINE = False
needs_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="sts2_rs wheel not importable")


@pytest.fixture(autouse=True)
def _v111(monkeypatch: pytest.MonkeyPatch) -> None:
    from ppo.e2e import evaluate as EV

    monkeypatch.setattr(sts2_client, "GAME_VERSION", V111)
    EV._process_stamps.cache_clear()  # another test of this process may have filled it


def _want() -> dict[str, Any]:
    """This process's values, each from its own source (not from `stamps.stamps`)."""
    return {
        "game_version": V111,
        "py_src": python_source_digest(),
        "py_client": ST.file_digest(Path(sts2_client.__file__)),
        "wheel": json.loads(json.dumps(wheel_identity())),
        "vocab_sha256": L.vocab_sha256(),
        "schema_sha256": C.schema_sha256(),
    }


def _check(st: Any, where: str) -> None:
    want = _want()
    assert set(want) == set(ST.STAMP_KEYS)
    assert set(OC.STAMP_FIELDS) <= set(ST.STAMP_KEYS)
    assert isinstance(st, dict), f"{where}: no stamps"
    got = {k: st.get(k, "<absent>") for k in ST.STAMP_KEYS}
    bad = {k: (got[k], want[k]) for k in ST.STAMP_KEYS if got[k] != want[k]}
    assert not bad, f"{where}: {bad}"


def test_cfg_stats_rows_and_every_checkpoint_carry_the_stamps(tmp_path: Path) -> None:
    it = TL.ITER
    cfg = TL.make_cfg(
        tmp_path,
        max_iters=4,
        state_every=it,
        full_every=2 * it,
        slim_every=2 * it,
        pin_every=2 * it,
    )
    lr = TL.make_learner(cfg)
    assert lr.train() == "max_iters"
    _check(json.loads((tmp_path / "cfg.json").read_text(encoding="utf-8"))["stamps"], "cfg.json")
    assert lr.stats_rows
    for r in lr.stats_rows:
        _check(r["stamps"], f"stats row {r['it']}")
    on_disk = (tmp_path / OC.STATS).read_text(encoding="utf-8").splitlines()
    assert len(on_disk) == len(lr.stats_rows)
    for line in on_disk:
        r = json.loads(line)
        _check(r["stamps"], f"stats.jsonl it {r['it']}")
    sd = ckpt.SaveDir(tmp_path)
    paths = [sd.state, *sd.full_dir.glob("*.pt"), *sd.pinned_dir.glob("*.pt")]
    paths += list(sd.slim_dir.glob("*.pt"))
    kinds = {p.parent.name for p in paths}
    assert {sd.full_dir.name, sd.pinned_dir.name, sd.slim_dir.name} <= kinds, kinds
    for p in paths:
        st = torch.load(p, map_location="cpu", weights_only=True)
        if "stamps" in st or p == sd.state:
            _check(st.get("stamps"), f"{p.name} top level")
        _check(st["net"]["cfg"]["stamps"], f"{p.name} net cfg")
    out = tmp_path / "raw.pt"
    export_variant(str(sd.state), "raw", str(out))
    _net, ncfg = load_checkpoint(out)
    _check(ncfg["stamps"], "exported student")


@needs_engine
def test_a_students_eval_row_carries_the_stamps(tmp_path: Path) -> None:
    """A row `evaluate.Player` plays with the student on the campaign env."""
    from ppo.e2e import evaluate as EV
    from ppo.e2e import net as N

    path = tmp_path / "student.pt"
    net = N.E2ENet(N.NetConfig.from_dict(TL.TINY), seed=0).eval()
    N.save_checkpoint(path, net, stamps={})
    player = EV.Player("ppo.e2e.env:E2EEnv", "ppo.e2e.net:make_student", str(path), "cpu")
    row = player.play(1_300_000_031)
    assert row["outcome"] in ("won", "lost", "capped"), row.get("error")
    _check(row, "eval row")


def test_an_eval_error_row_carries_the_stamps() -> None:
    from ppo.e2e import evaluate as EV

    row = EV.error_row(1_300_000_031, "ppo.e2e.net:make_student", "boom")
    assert (row["outcome"], row["won"], row["error"]) == ("error", False, "boom")
    _check(row, "eval error row")


# The e2e lane is Linux-only (systemd units, /dev/shm); the public CI also runs Windows.
pytestmark = pytest.mark.skipif(
    'sys.platform == "win32"',
    reason="the stamps test runs the learner, which needs POSIX shared memory (see test_learner.py). Linux only.",
)
