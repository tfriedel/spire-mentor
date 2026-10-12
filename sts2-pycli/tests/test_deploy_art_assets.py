"""The two generated art files must exist wherever something COPYs or reads them.

`deploy/asset_manifest.json` + `deploy/creature_scales.json` are gitignored
build artifacts — they go stale the moment the art tree changes, so a fresh
checkout simply does not have them and whatever consumes them dies on a missing
file. That is not hypothetical: the Cloud Run image build failed exactly this
way at its `COPY`, which is why the generator became a dependency of the recipes
that need it.

Cloud Run was retired on 2026-09-08 (service deleted, its recipes and
deploy/*.sh removed), so the bucket-sync half of this module went with it: there
is no longer a public art bucket in the serving path, because the static build
hard-links the art into `web/dist` instead. What remains is the generator
contract for the two live consumers — the static art step and the root
`Dockerfile`, which is still how the server-backed workbench runs locally.
"""

import re
from pathlib import Path

import pytest

_PYCLI_ROOT = Path(__file__).resolve().parents[1]
_MONO_ROOT = _PYCLI_ROOT.parent
_JUSTFILE = _PYCLI_ROOT / "justfile"
_DOCKERFILE = _MONO_ROOT / "Dockerfile"

# The generator that writes both files, and what it writes.
_GENERATOR = "scripts/build_asset_manifest.py"
_ARTIFACTS = ["asset_manifest.json", "creature_scales.json"]


def _justfile() -> str:
    return _JUSTFILE.read_text(encoding="utf-8")


@pytest.mark.parametrize("artifact", _ARTIFACTS)
def test_dockerfile_copies_the_generated_artifacts(artifact: str) -> None:
    """Establishes the premise: the image really does need these files."""
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(rf"^COPY .*{re.escape(artifact)}", text, re.M), (
        f"{artifact} is no longer COPYd into the image — if that is deliberate, "
        f"this whole test module is obsolete"
    )


@pytest.mark.parametrize("artifact", _ARTIFACTS)
def test_generated_artifacts_are_gitignored(artifact: str) -> None:
    """They are per-machine build output; committing one is how it goes stale."""
    ignored = (_PYCLI_ROOT / "deploy" / ".gitignore").read_text(encoding="utf-8")
    assert artifact in ignored


def test_static_art_step_runs_the_generator() -> None:
    """`_art-into-dist` is the live consumer: it reads the manifest it builds.

    `scripts/art_to_dist.py` refuses to finish unless every file the manifest
    names made it into `web/dist/assets`, so a stale or absent manifest is the
    difference between a complete static deploy and a silently thin one.
    """
    text = _justfile()
    assert re.search(r"^_art-into-dist:", text, re.M), "_art-into-dist recipe is missing"
    body = text.split("_art-into-dist:", 1)[1].split("\n\n", 1)[0]
    # The recipe delegates to the cross-platform build driver's `art` stage.
    assert "scripts/workbench_static.py art" in body, "_art-into-dist must run the `art` stage"
    driver = (_PYCLI_ROOT / "scripts" / "workbench_static.py").read_text(encoding="utf-8")
    art_stage = driver.split("\ndef art(", 1)[1].split("\ndef ", 1)[0]
    assert _GENERATOR in art_stage, f"the `art` stage must run {_GENERATOR}"


def test_no_recipe_still_syncs_art_to_the_retired_bucket() -> None:
    """The bucket left the serving path with Cloud Run; nothing may push to it."""
    text = _justfile()
    for gone in ("deploy-art:", "check-art-bucket:", "_workbench-assets:"):
        assert not re.search(rf"^{re.escape(gone)}", text, re.M), (
            f"{gone} came back — the art bucket is not in the serving path any more "
            f"(and billing is off on the project, so it 403s)"
        )


def test_art_to_dist_refuses_git_lfs_pointer_files(tmp_path: Path) -> None:
    """The wiki's art and audio are LFS-tracked. A checkout that never smudged
    them holds ~130-byte pointer text files under the real names, and the
    packer used to ship those verbatim: the 2026-09-08 deploy served pointers
    for the whole audio tree, so every sound in the browser was silent with
    no error anywhere but the devtools console. The packer must refuse."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "art_to_dist", _PYCLI_ROOT / "scripts" / "art_to_dist.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    images, audio = tmp_path / "images", tmp_path / "audio"
    (audio / "sfx").mkdir(parents=True)
    images.mkdir()
    (audio / "sfx" / "real.webm").write_bytes(b"\x1aE\xdf\xa3webm-bytes")
    (audio / "sfx" / "block_gain.webm").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:5a6094a309763c81b3e82873395ba9fdbf04c5b2cdf4a25b9bb6af67fdbb91a1\n"
        "size 6225\n"
    )
    files = ["audio/sfx/real.webm", "audio/sfx/block_gain.webm"]
    with pytest.raises(
        SystemExit, match=r"LFS POINTERS.*block_gain.*git -C \.\./sts2-wiki lfs pull"
    ):
        mod.convert_tree(
            files,
            images=images,
            audio=audio,
            dest=tmp_path / "dist",
            cache=tmp_path / "cache",
            jobs=1,
        )
    # Real bytes alone go through (no PNG in the tree, so no cwebp needed).
    shipped = mod.convert_tree(
        files[:1],
        images=images,
        audio=audio,
        dest=tmp_path / "dist",
        cache=tmp_path / "cache",
        jobs=1,
    )
    assert shipped == {"audio/sfx/real.webm": 14}
