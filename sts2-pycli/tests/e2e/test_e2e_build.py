"""The e2e lane build (GH #1535, `ppo.e2e.build`): one switch for every v0.111.0 hardcode.

A process's lane build comes from `STS2_E2E_GAME_VERSION` at import; unset it is
v0.111.0, so these tests (run without it) check that the default lane reads exactly what
it always did, and check the later build's tables in a subprocess that sets it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from ppo.e2e import build, factory
from ppo.e2e import contracts as C
from ppo.e2e import vocab as V

PY_DIR = Path(__file__).resolve().parents[2] / "python"


def test_the_default_lane_is_v0111() -> None:
    assert build.GAME_VERSION == build.DEFAULT_BUILD == "v0.111.0"
    assert C.GAME_VERSION == factory.GAME_VERSION == "v0.111.0"
    assert factory.CHARACTERS == ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")
    assert V.TSV_PATH == V.BASE_TSV_PATH and V.TSV_PATH.name == "vocab.tsv"


def test_resolve_names_a_build_or_refuses() -> None:
    assert build.resolve(None) == build.resolve("") == "v0.111.0"
    assert build.resolve("v0.113.0") == build.resolve("0.113.1") == "v0.113.0"
    with pytest.raises(ValueError):
        build.resolve("v0.107.1")  # a supported game build, but no e2e lane plays it
    with pytest.raises(ValueError):
        build.resolve("v0.115.0")
    assert build.resolve("v0.114.0") == "v0.113.0"  # sts2_client.PLAYS_AS


def test_the_fool_plays_from_v0113_on() -> None:
    assert "Fool" not in build.characters("v0.111.0")
    assert build.characters("v0.113.0") == (*build.characters("v0.111.0"), "Fool")
    # The schema's character ids already name it (CHAR_ID 6).
    assert build.characters("v0.113.0") == C.CHARACTERS


def test_a_later_builds_vocab_extends_vocab_tsv() -> None:
    """Append-only across builds: every `vocab.tsv` row keeps its id in the later
    build's table, so a v0.111.0 net's rows mean the same there."""
    path = V.tsv_path("v0.113.0")
    assert path.name == "vocab_v0113.tsv"
    if not path.exists():
        pytest.skip("vocab_v0113.tsv not generated yet")
    base, later = V.read_tsv(V.BASE_TSV_PATH), V.read_tsv(path)
    assert later[: len(base)] == base
    new = later[len(base) :]
    assert new and all(r.id >= len(base) for r in new)
    names = {(r.registry, r.name) for r in new}
    assert ("card", "STRIKE_FOOL") in names and ("card", "DEFEND_FOOL") in names
    assert max(r.id for r in later) < C.VOCAB_ROWS


def _later(code: str) -> dict:
    env = {
        **os.environ,
        build.BUILD_ENV: "v0.113.0",
        "STS2_GAME_VERSION": "v0.113.0",
        "PYTHONPATH": str(PY_DIR),
    }
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_v0113_process_reads_the_later_build_everywhere() -> None:
    got = _later(
        "import json\n"
        "from ppo.e2e import build, contracts as C, factory, evaluate, invariants, vocab as V\n"
        "from ppo.e2e import ops_contract as OC, decomp\n"
        "print(json.dumps({'b': build.GAME_VERSION, 'c': C.GAME_VERSION, 'f': factory.GAME_VERSION,"
        " 'e': evaluate.GAME_VERSION, 'i': invariants.EXPECTED_GAME_VERSION,"
        " 'oc': OC.LANE_ENVIRONMENT['STS2_GAME_VERSION'], 'ocb': OC.LANE_ENVIRONMENT[build.BUILD_ENV],"
        " 'chars': list(factory.CHARACTERS), 'tsv': V.TSV_PATH.name, 'decomp': decomp.DECOMP_VERSION,"
        " 'schema': C.schema_sha256()}))"
    )
    assert {got[k] for k in ("b", "c", "f", "e", "i", "oc", "ocb")} == {"v0.113.0"}
    assert got["chars"][-1] == "Fool"
    assert got["tsv"] == "vocab_v0113.tsv"
    assert got["decomp"] == "v0.113.1"
    # The build is part of the schema: a v0.113.0 lane never loads a v0.111.0 net as is.
    assert got["schema"] != C.schema_sha256()


def test_a_v0111_lanes_schema_did_not_move() -> None:
    """The default lane's schema hash: the one every #1097/#1197 checkpoint carries."""
    assert C.schema_spec()["game_version"] == "v0.111.0"
    assert C.schema_sha256() == "7afdc933284d80e400138267d46e51e2625a3ff09671d1028b131300c09f7f79"


RS_DATA = Path(__file__).resolve().parents[3] / "sts2-rs" / "crates" / "sts2-e2e" / "data"


@pytest.mark.parametrize("build_", ["v0.111.0", "v0.113.0"])
def test_the_rust_tokenizer_embeds_each_builds_vocab_verbatim(build_: str) -> None:
    """`sts2-e2e/data/vocab_<tag>.tsv` is the Python table byte for byte (the Rust
    tokenizer of that lane build reads it; `src/vocab.rs`)."""
    tag = "v0111" if build_ == "v0.111.0" else "v0113"
    assert (RS_DATA / f"vocab_{tag}.tsv").read_bytes() == V.tsv_path(build_).read_bytes()


def test_the_rust_tables_of_a_v0113_lane_match_python() -> None:
    """Card types and HITS: the copies `sts2-e2e/data/*_v0113.tsv` hold."""
    got = _later(
        "import json\n"
        "import numpy as np\n"
        "from ppo.e2e import content_prior as CP\n"
        "from ppo.combat_ctx import static_tables as ST\n"
        "h = CP.table()[:, CP.C['HITS']]\n"
        "print(json.dumps({'types': ST.card_types('v0.113.0'),"
        " 'hits': {str(i): float(np.float32(h[i])) for i in np.flatnonzero(h)}}))"
    )

    def rows(name: str) -> list[list[str]]:
        text = (RS_DATA / name).read_text(encoding="utf-8")
        return [ln.split("\t") for ln in text.splitlines() if ln and not ln.startswith("#")]

    assert dict(rows("card_types_v0113.tsv")) == got["types"]
    assert {i: float(v) for i, v in rows("hits_v0113.tsv")} == pytest.approx(got["hits"])


def test_migrate_carries_a_v0111_net_to_v0113_with_twinned_starters() -> None:
    """A fresh v0.111.0 net migrates: v0.113.0 stamps, old rows untouched, the Fool's
    Strike row = the mean of the five starters' Strike rows."""
    got = _later(
        "import json, torch\n"
        "from ppo.e2e import contracts as C, migrate as M, net as N, vocab as V\n"
        "net = N.E2ENet(N.NetConfig(n_layers=1), seed=0)\n"
        "base = M._base_vocab()\n"
        "src = N.net_state(net, schema_sha256=M._base_schema_sha256(),"
        " vocab_sha256=base.sha256, stamps={'game_version': 'v0.111.0', 'filled': []})\n"
        "out = M.migrate(src, filled=[], twins=M.DEFAULT_TWINS)\n"
        "v = V.load(); w0, w1 = src['state_dict']['ids.weight'], out['state_dict']['ids.weight']\n"
        "sf = v.id('card', 'STRIKE_FOOL'); olds = [v.id('card', f'STRIKE_{c}') for c in M.STARTERS]\n"
        "print(json.dumps({'schema': out['cfg']['schema_sha256'] == C.schema_sha256(),"
        " 'gv': out['cfg']['stamps']['game_version'],"
        " 'old_same': bool(torch.equal(w0[: base.n_used], w1[: base.n_used])),"
        " 'twin': bool(torch.allclose(w1[sf], w0[olds].mean(0)))}))"
    )
    assert got == {"schema": True, "gv": "v0.113.0", "old_same": True, "twin": True}
