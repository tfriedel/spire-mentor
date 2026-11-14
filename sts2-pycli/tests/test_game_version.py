"""GH #789: per-version C# oracle selection (``STS2_GAME_VERSION``).

The oracle is built once per supported game build — DLLs in ``lib/<ver>/``,
binary in ``bin/<ver>/`` — and every Python entry point resolves which one to
drive through ``sts2_client``. These tests pin that resolution, the trace-cache
namespace split, and the guard that keeps a v0.107.1 call site byte-for-byte
the call it was.
"""

from __future__ import annotations

import os
import time

import parity_audit as pa
import pytest
import sts2_client


@pytest.fixture
def restore_version():
    """Undo any ``set_game_version`` a test does (it is process-global)."""
    before = sts2_client.GAME_VERSION
    env = os.environ.get(sts2_client.GAME_VERSION_ENV)
    yield
    sts2_client.set_game_version(before)
    if env is None:
        os.environ.pop(sts2_client.GAME_VERSION_ENV, None)
    else:
        os.environ[sts2_client.GAME_VERSION_ENV] = env


def test_the_default_and_the_legacy_build_are_two_things() -> None:
    """Naming no build gets v0.111.0; a file that names none IS v0.107.1."""
    assert sts2_client.DEFAULT_GAME_VERSION == "v0.111.0"
    assert sts2_client.LEGACY_GAME_VERSION == "v0.107.1"
    assert sts2_client.SUPPORTED_GAME_VERSIONS[0] == sts2_client.LEGACY_GAME_VERSION
    assert sts2_client.DEFAULT_GAME_VERSION in sts2_client.SUPPORTED_GAME_VERSIONS


@pytest.mark.parametrize("raw", ["v0.111.0", "0.111.0", " v0.111.0 "])
def test_normalize_accepts_both_spellings(raw) -> None:
    assert sts2_client.normalize_game_version(raw) == "v0.111.0"


def test_normalize_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match="unsupported game version"):
        sts2_client.normalize_game_version("v0.103.2")


def test_resolve_reads_env(monkeypatch) -> None:
    monkeypatch.delenv(sts2_client.GAME_VERSION_ENV, raising=False)
    assert sts2_client.resolve_game_version() == "v0.111.0"
    monkeypatch.setenv(sts2_client.GAME_VERSION_ENV, "0.107.1")
    assert sts2_client.resolve_game_version() == "v0.107.1"


def test_paths_are_version_keyed() -> None:
    # The apphost's file name is per-platform and pinned by
    # `test_headless_bin_names_the_platform_apphost`; this pins the directory.
    root = sts2_client.REPO_ROOT
    assert os.path.dirname(sts2_client.headless_bin("v0.111.0")) == os.path.join(
        root, "src", "Sts2Headless", "bin", "v0.111.0", "Debug", "net9.0"
    )
    assert sts2_client.lib_dir("v0.111.0") == os.path.join(root, "lib", "v0.111.0")


@pytest.mark.parametrize(
    ("platform", "apphost"), [("win32", "Sts2Headless.exe"), ("linux", "Sts2Headless")]
)
def test_headless_bin_names_the_platform_apphost(monkeypatch, platform, apphost) -> None:
    """GH #506: the Windows apphost is ``Sts2Headless.exe``.

    Without the suffix ``_spawn_cmd``'s ``isfile`` probe never matched there,
    so every spawn took the slow ``dotnet run`` fallback with the binary built.
    """
    monkeypatch.setattr(sts2_client.sys, "platform", platform)
    assert os.path.basename(sts2_client.headless_bin("v0.107.1")) == apphost


def test_set_game_version_moves_globals_and_env(restore_version) -> None:
    sts2_client.set_game_version("0.111.0")
    assert sts2_client.GAME_VERSION == "v0.111.0"
    assert os.environ[sts2_client.GAME_VERSION_ENV] == "v0.111.0"
    assert sts2_client.headless_bin("v0.111.0") == sts2_client.HEADLESS_BIN
    assert sts2_client.lib_dir("v0.111.0") == sts2_client.LIB_DIR
    # The `dotnet run` fallback must ask MSBuild for the same build.
    os.environ.pop("STS2_HEADLESS_BIN", None)
    if not os.path.isfile(sts2_client.HEADLESS_BIN):
        assert "--property:GameVersion=v0.111.0" in sts2_client._spawn_cmd()


def test_legacy_flat_lib_fallback_is_v0107_only(tmp_path, monkeypatch) -> None:
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "sts2.dll").write_bytes(b"MZ")
    monkeypatch.setattr(sts2_client, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(sts2_client, "_LEGACY_LIB_WARNED", False)
    with pytest.warns(DeprecationWarning, match="legacy flat game-DLL layout"):
        assert sts2_client.lib_dir("v0.107.1") == str(tmp_path / "lib")
    # Never for another build: the flat layout only ever held v0.107.1.
    assert sts2_client.lib_dir("v0.111.0") == str(tmp_path / "lib" / "v0.111.0")
    # Once lib/<ver>/ exists it wins.
    (tmp_path / "lib" / "v0.107.1").mkdir()
    (tmp_path / "lib" / "v0.107.1" / "sts2.dll").write_bytes(b"MZ")
    assert sts2_client.lib_dir("v0.107.1") == str(tmp_path / "lib" / "v0.107.1")


def test_non_default_oracle_needs_its_own_lib(tmp_path, monkeypatch) -> None:
    # A Steam install (STS2_GAME_DIR) can stand in for the default build only.
    game = tmp_path / "steam"
    game.mkdir()
    (game / "sts2.dll").write_bytes(b"MZ")
    monkeypatch.setattr(sts2_client, "REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("STS2_GAME_DIR", str(game))
    assert sts2_client.oracle_available("v0.107.1")
    assert not sts2_client.oracle_available("v0.111.0")
    lib = tmp_path / "lib" / "v0.111.0"
    lib.mkdir(parents=True)
    (lib / "sts2.dll").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    assert not sts2_client.oracle_available("v0.111.0")  # an LFS pointer is no assembly
    (lib / "sts2.dll").write_bytes(b"MZ\x90\x00")
    assert sts2_client.oracle_available("v0.111.0")


def test_rs_engine_call_always_passes_the_version(restore_version) -> None:
    seen = {}

    def ctor(*, string_seed, game_version=None):
        seen.update(string_seed=string_seed, game_version=game_version)
        return "rm"

    # The default build is passed explicitly too: `game_version="v0.107.1"` is
    # the engine's default, so a v0.107.1 call simulates what it always did.
    sts2_client.set_game_version("v0.107.1")
    assert sts2_client.rs_engine_call(ctor, string_seed="S") == "rm"
    assert seen == {"string_seed": "S", "game_version": "v0.107.1"}
    sts2_client.set_game_version("v0.111.0")
    sts2_client.rs_engine_call(ctor, string_seed="S")
    assert seen == {"string_seed": "S", "game_version": "v0.111.0"}
    # An explicit `version=` wins over the process-wide selection.
    sts2_client.rs_engine_call(ctor, string_seed="S", version="0.107.1")
    assert seen == {"string_seed": "S", "game_version": "v0.107.1"}


def test_rs_engine_call_has_no_old_wheel_fallback(restore_version) -> None:
    # A wheel without `game_version=` must fail loudly: silently simulating
    # v0.107.1 against a v0.111.0 oracle turns every divergence into skew.
    def ctor(**kw):
        if "game_version" in kw:
            raise TypeError("RunManager.__new__() got an unexpected keyword argument")
        return "rm"

    sts2_client.set_game_version("v0.111.0")
    with pytest.raises(TypeError, match="unexpected keyword"):
        sts2_client.rs_engine_call(ctor, string_seed="S")


def test_installed_engine_takes_game_version_everywhere() -> None:
    sts2_rs = pytest.importorskip("sts2_rs")
    for ver in sts2_client.SUPPORTED_GAME_VERSIONS:
        rm = sts2_client.rs_engine_call(
            sts2_rs.RunManager, string_seed="ABC", character="Ironclad", version=ver
        )
        assert rm.game_version == ver
        assert sts2_client.rs_engine_call(sts2_rs.roll_acts, "ABC", version=ver)


def test_rs_engine_call_reraises_unrelated_type_errors(restore_version) -> None:
    def ctor(**kw):
        raise TypeError("character must be a str")

    sts2_client.set_game_version("v0.111.0")
    with pytest.raises(TypeError, match="character must be a str"):
        sts2_client.rs_engine_call(ctor, string_seed="S")


def _oracle_tree(root, version: str, dll_bytes: bytes = b"GAME") -> None:
    lib = root / "lib" / version
    lib.mkdir(parents=True)
    (lib / "sts2.dll").write_bytes(dll_bytes)
    src = root / "src" / "Sts2Headless"
    src.mkdir(parents=True, exist_ok=True)
    (src / "RunSimulator.cs").write_text("class RunSim {}")


def test_namespace_is_prefixed_and_split_by_version(tmp_path) -> None:
    _oracle_tree(tmp_path, "v0.107.1")
    _oracle_tree(tmp_path, "v0.111.0")
    a = pa.cs_version_namespace(str(tmp_path), "v0.107.1")
    b = pa.cs_version_namespace(str(tmp_path), "v0.111.0")
    assert a.startswith("v0.107.1-") and b.startswith("v0.111.0-")
    # Same bytes in both lib dirs, yet two builds never share a cache.
    assert a.split("-", 1)[1] != b.split("-", 1)[1]


def test_namespace_hashes_only_that_versions_lib(tmp_path) -> None:
    _oracle_tree(tmp_path, "v0.107.1")
    _oracle_tree(tmp_path, "v0.111.0", dll_bytes=b"V1")
    before = pa.cs_version_namespace(str(tmp_path), "v0.107.1")
    (tmp_path / "lib" / "v0.111.0" / "sts2.dll").write_bytes(b"V2")
    assert pa.cs_version_namespace(str(tmp_path), "v0.107.1") == before


def test_oracle_fresh_checks_the_selected_versions_build(tmp_path) -> None:
    src = tmp_path / "src" / "Sts2Headless"
    src.mkdir(parents=True)
    cs = src / "RunSimulator.cs"
    cs.write_text("// source")
    built = src / "bin" / "v0.107.1" / "Debug" / "net9.0"
    built.mkdir(parents=True)
    (built / "Sts2Headless.dll").write_text("binary")
    old, new = time.time() - 100, time.time()
    os.utime(cs, (old, old))
    os.utime(built / "Sts2Headless.dll", (new, new))
    pa.assert_oracle_fresh(str(tmp_path), "v0.107.1")  # fresh: no raise
    # A fresh v0.107.1 build cannot vouch for an unbuilt v0.111.0 one.
    with pytest.raises(SystemExit, match=r"NOT BUILT for v0\.111\.0"):
        pa.assert_oracle_fresh(str(tmp_path), "v0.111.0")
