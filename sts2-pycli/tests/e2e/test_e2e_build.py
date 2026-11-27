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
    assert build.resolve("v0.114.0") == build.resolve("0.114.2") == "v0.114.0"


def test_each_later_build_warm_starts_from_the_one_before() -> None:
    assert build.previous("v0.113.0") == "v0.111.0"
    assert build.previous("v0.114.0") == "v0.113.0"
    with pytest.raises(ValueError):
        build.previous("v0.111.0")
    assert set(build.CARD_KIND) == set(build.BUILDS)


def test_the_fool_plays_from_v0113_on() -> None:
    assert "Fool" not in build.characters("v0.111.0")
    assert build.characters("v0.113.0") == (*build.characters("v0.111.0"), "Fool")
    # The schema's character ids already name it (CHAR_ID 6).
    assert build.characters("v0.113.0") == C.CHARACTERS
    assert build.characters("v0.114.0") == C.CHARACTERS


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


def test_the_v0114_vocab_extends_v0113s() -> None:
    """`vocab_v0114.tsv` is `vocab_v0113.tsv`'s rows unchanged, then v0.114.0's two new
    cards; Rapid Shot, which v0.114.0 removed, keeps its row."""
    prev, path = V.tsv_path("v0.113.0"), V.tsv_path("v0.114.0")
    assert path.name == "vocab_v0114.tsv"
    old, new = V.read_tsv(prev), V.read_tsv(path)
    assert new[: len(old)] == old
    assert [(r.id, r.registry, r.name) for r in new[len(old) :]] == [
        (2136, "card", "OPTIMISM"),
        (2137, "card", "TWIN_SHOT"),
    ]
    assert V.Vocab(new).has("card", "RAPID_SHOT")


def _later(code: str, build_: str = "v0.113.0") -> dict:
    env = {
        **os.environ,
        build.BUILD_ENV: build_,
        "STS2_GAME_VERSION": build_,
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


def test_a_v0114_process_reads_the_later_build_everywhere() -> None:
    got = _later(
        "import json\n"
        "from ppo.e2e import build, contracts as C, factory, evaluate, invariants, vocab as V\n"
        "from ppo.e2e import ops_contract as OC, decomp, content_prior as CP\n"
        "from ppo.combat_ctx import tokens as T\n"
        "print(json.dumps({'b': build.GAME_VERSION, 'c': C.GAME_VERSION, 'f': factory.GAME_VERSION,"
        " 'e': evaluate.GAME_VERSION, 'i': invariants.EXPECTED_GAME_VERSION,"
        " 'oc': OC.LANE_ENVIRONMENT['STS2_GAME_VERSION'], 'ocb': OC.LANE_ENVIRONMENT[build.BUILD_ENV],"
        " 'chars': list(factory.CHARACTERS), 'tsv': V.TSV_PATH.name, 'decomp': decomp.DECOMP_VERSION,"
        " 'prior': CP.TSV.name, 'prior_ok': CP.sha256() == CP.TABLE_SHA256,"
        " 'sizes': T.vocab().digest() == dict(C.SCTX_TABLE_SIZES), 'card': T._kinds()['card']}))",
        "v0.114.0",
    )
    assert {got[k] for k in ("b", "c", "f", "e", "i", "oc", "ocb")} == {"v0.114.0"}
    assert got["chars"][-1] == "Fool"
    assert got["tsv"] == "vocab_v0114.tsv"
    assert got["prior"] == "content_prior_v0114.tsv" and got["prior_ok"]
    assert got["decomp"] == "v0.114.0"
    assert got["sizes"] and got["card"] == "card_v0114"


# The v0.113.0 lane's stamps as the Fool lane trained them (before v0.114.0 had a lane):
# adding v0.114.0 must not move one of them, or the running lane's checkpoints refuse.
V0113_STAMPS = {
    "schema": "406679123c041aecf200a366d6b0e8445a63c384f413e6a0fbd645b5a0c1df58",
    "vocab": "4420b60ba36454049e034f15c7aa20ca3df34c9aee067dd5daab1230dfe23bea",
    "prior": "a61db28c479c56b4c017746d937d0994ab9f2a060a0c0260eeb46e488baf1054",
    "card": "card_v0113",
}


def test_a_v0113_lanes_stamps_did_not_move() -> None:
    got = _later(
        "import json\n"
        "from ppo.e2e import contracts as C, vocab as V, content_prior as CP\n"
        "from ppo.combat_ctx import tokens as T\n"
        "assert T.vocab().digest() == dict(C.SCTX_TABLE_SIZES)\n"
        "print(json.dumps({'schema': C.schema_sha256(), 'vocab': V.load().sha256,"
        " 'prior': CP.sha256(), 'card': T._kinds()['card']}))"
    )
    assert got == V0113_STAMPS


@pytest.mark.parametrize("build_", ["v0.113.0", "v0.114.0"])
def test_a_later_builds_sctx_sizes_match_its_tables(build_: str) -> None:
    """`contracts.SCTX_TABLE_SIZES` is what `tokens.vocab()` builds on that build."""
    got = _later(
        "import json\n"
        "from ppo.e2e import contracts as C\n"
        "from ppo.combat_ctx import tokens as T\n"
        "print(json.dumps({'got': T.vocab().digest(), 'want': dict(C.SCTX_TABLE_SIZES)}))",
        build_,
    )
    assert got["got"] == got["want"] == dict(C.SCTX_TABLE_SIZES_BY_BUILD[build_])


def test_a_v0111_lanes_schema_did_not_move() -> None:
    """The default lane's schema hash: the one every #1097/#1197 checkpoint carries."""
    assert C.schema_spec()["game_version"] == "v0.111.0"
    assert C.schema_sha256() == "7afdc933284d80e400138267d46e51e2625a3ff09671d1028b131300c09f7f79"


RS_DATA = Path(__file__).resolve().parents[3] / "sts2-rs" / "crates" / "sts2-e2e" / "data"


def _tag(build_: str) -> str:
    major, minor, _patch = build_.removeprefix("v").split(".")
    return f"v{major}{minor}"


@pytest.mark.parametrize("build_", ["v0.111.0", "v0.113.0", "v0.114.0"])
def test_the_rust_tokenizer_embeds_each_builds_vocab_verbatim(build_: str) -> None:
    """`sts2-e2e/data/vocab_<tag>.tsv` is the Python table byte for byte (the Rust
    tokenizer of that lane build reads it; `src/vocab.rs`)."""
    assert (RS_DATA / f"vocab_{_tag(build_)}.tsv").read_bytes() == V.tsv_path(build_).read_bytes()


@pytest.mark.parametrize("build_", ["v0.113.0", "v0.114.0"])
def test_the_rust_tables_of_a_later_lane_match_python(build_: str) -> None:
    """Card types and HITS: the copies `sts2-e2e/data/*_<tag>.tsv` hold."""
    got = _later(
        "import json\n"
        "import numpy as np\n"
        "from ppo.e2e import content_prior as CP\n"
        "from ppo.combat_ctx import static_tables as ST\n"
        "h = CP.table()[:, CP.C['HITS']]\n"
        f"print(json.dumps({{'types': ST.card_types({build_!r}),"
        " 'hits': {str(i): float(np.float32(h[i])) for i in np.flatnonzero(h)}}))",
        build_,
    )

    def rows(name: str) -> list[list[str]]:
        text = (RS_DATA / name).read_text(encoding="utf-8")
        return [ln.split("\t") for ln in text.splitlines() if ln and not ln.startswith("#")]

    tag = _tag(build_)
    assert dict(rows(f"card_types_{tag}.tsv")) == got["types"]
    assert {i: float(v) for i, v in rows(f"hits_{tag}.tsv")} == pytest.approx(got["hits"])


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


def test_migrate_carries_a_v0113_net_to_v0114_with_twinned_cards() -> None:
    """A fresh v0.113.0 net migrates to v0.114.0: v0.114.0 stamps, every v0.113.0 row
    untouched (Rapid Shot's too), Twin Shot = Rapid Shot's row, Optimism = the mean of
    Growth's and Punch of Joy's; a v0.111.0 net does not migrate there."""
    got = _later(
        "import json, torch\n"
        "from ppo.e2e import contracts as C, migrate as M, net as N, vocab as V\n"
        "net = N.E2ENet(N.NetConfig(n_layers=1), seed=0)\n"
        "base = M._base_vocab()\n"
        "st = {'game_version': 'v0.113.0', 'filled': ['chars', 'fool']}\n"
        "src = N.net_state(net, schema_sha256=M._base_schema_sha256(),"
        " vocab_sha256=base.sha256, stamps=st)\n"
        "out = M.migrate(src, filled=['chars', 'fool'], twins=M.DEFAULT_TWINS)\n"
        "v = V.load(); w0, w1 = src['state_dict']['ids.weight'], out['state_dict']['ids.weight']\n"
        "i = lambda n: v.id('card', n)\n"
        "bad = []\n"
        "for gv in ('v0.111.0', None):\n"
        "    s2 = N.net_state(net, schema_sha256=M._base_schema_sha256(),"
        " vocab_sha256=base.sha256, stamps={'game_version': gv, 'filled': []})\n"
        "    try:\n"
        "        M.migrate(s2, filled=[], twins={})\n"
        "    except M.MigrateError:\n"
        "        bad.append(gv)\n"
        "print(json.dumps({'schema': out['cfg']['schema_sha256'] == C.schema_sha256(),"
        " 'vocab': out['cfg']['vocab_sha256'] == v.sha256,"
        " 'gv': out['cfg']['stamps']['game_version'],"
        " 'from': out['cfg']['provenance']['from_build'],"
        " 'old_same': bool(torch.equal(w0[: base.n_used], w1[: base.n_used])),"
        " 'twin': bool(torch.equal(w1[i('TWIN_SHOT')], w0[i('RAPID_SHOT')])),"
        " 'opt': bool(torch.allclose(w1[i('OPTIMISM')],"
        " w0[[i('GROWTH'), i('PUNCH_OF_JOY')]].mean(0))),"
        " 'twins': out['cfg']['provenance']['twins'], 'refused': bad}))",
        "v0.114.0",
    )
    assert got == {
        "schema": True,
        "vocab": True,
        "gv": "v0.114.0",
        "from": "v0.113.0",
        "old_same": True,
        "twin": True,
        "opt": True,
        "twins": {"TWIN_SHOT": ["RAPID_SHOT"], "OPTIMISM": ["GROWTH", "PUNCH_OF_JOY"]},
        "refused": ["v0.111.0", None],
    }


def test_migrate_rebuilds_the_source_builds_schema() -> None:
    """`migrate._base_schema_sha256` in a v0.114.0 process is a v0.113.0 lane's own
    `schema_sha256`, so a real v0.113.0 checkpoint passes the source check."""
    a = _later(
        "import json\nfrom ppo.e2e import contracts as C\nprint(json.dumps(C.schema_sha256()))"
    )
    b = _later(
        "import json\nfrom ppo.e2e import migrate as M\nprint(json.dumps(M._base_schema_sha256()))",
        "v0.114.0",
    )
    assert a == b
