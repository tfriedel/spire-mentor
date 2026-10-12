#!/usr/bin/env python3
"""Tests for `extract_images.py`'s pck-stub resolution and the big-icon pass.

The module had no tests at all, which is how `_CTEX_REF`'s predecessor — a
bare `path="res://…"` — sat in place resolving 5 of 263 power stubs and 5 of
319 relic stubs without anyone noticing. A silent under-count is exactly the
failure a test has to catch here, because the pass still "succeeds" and still
writes files; there are just almost none of them.
"""

import os
import tempfile
import unittest
from pathlib import Path

from scripts.extract_images import _ctex_path, extract_power_big_icons, write_ownerless_sprites
from scripts.extract_rig import read_pck_index

#: A VRAM-compressed import stub, verbatim in shape from
#: `images/powers/accelerant_power.png.import` in the v0.107.1 pck. Powers and
#: relics all look like this; potions do not, which is why the narrower regex
#: passed for potions and failed everywhere else.
_VRAM_STUB = b"""[remap]

importer="texture"
type="CompressedTexture2D"
uid="uid://bklnfhfftn5dc"
path.s3tc="res://.godot/imported/accelerant_power.png-c2535fe4.s3tc.ctex"
metadata={
"imported_formats": ["s3tc_bptc", "etc2_astc"],
"vram_texture": true
}
"""

#: A lossless import, which writes a single unsuffixed `path=`.
_LOSSLESS_STUB = b"""[remap]

importer="texture"
type="CompressedTexture2D"
path="res://.godot/imported/potion_x.png-deadbeef.ctex"
"""


class CtexPathTest(unittest.TestCase):
    def test_reads_a_vram_compressed_stub(self) -> None:
        # The regression: `path.s3tc=` is still a path, and it is the ONLY one
        # a VRAM-imported texture has.
        self.assertEqual(
            _ctex_path(_VRAM_STUB),
            ".godot/imported/accelerant_power.png-c2535fe4.s3tc.ctex",
        )

    def test_still_reads_a_plain_stub(self) -> None:
        self.assertEqual(_ctex_path(_LOSSLESS_STUB), ".godot/imported/potion_x.png-deadbeef.ctex")

    def test_returns_none_when_there_is_no_ctex_at_all(self) -> None:
        # Absent beats wrong: the caller skips the sprite rather than writing
        # a broken one.
        self.assertIsNone(_ctex_path(b'[remap]\nimporter="keep"\n'))


class ExtractPowerBigIconsTest(unittest.TestCase):
    """The pass itself, driven with a stub reader rather than the real pck so
    it runs everywhere — the pck-backed assertions live in the gate below."""

    def _run(self, index: dict[str, tuple[int, int]], files: dict[str, bytes]) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            extract_power_big_icons(index, lambda p: files.get(p), tmp)
            out = Path(tmp) / "power_big"
            return sorted(p.name for p in out.iterdir()) if out.is_dir() else []

    def test_skips_a_nested_folder_under_the_powers_root(self) -> None:
        # Only flat `images/powers/*.png.import` are power icons; a nested
        # directory is popup chrome, the same rule `extract_fallback_sprites`
        # applies to `images/potions/potion_pop_up/`.
        index = {"images/powers/popup/frame.png.import": (0, 0)}
        self.assertEqual(self._run(index, {"images/powers/popup/frame.png.import": _VRAM_STUB}), [])

    def test_skips_a_stub_whose_ctex_is_missing_rather_than_failing(self) -> None:
        index = {"images/powers/ghost_power.png.import": (0, 0)}
        self.assertEqual(self._run(index, {"images/powers/ghost_power.png.import": _VRAM_STUB}), [])


def _resolve_pck() -> Path:
    """The vendored pck, honouring `STS2_PCK` — same walk-up as
    `test_extract_vfx._resolve_pck`, so it works from a worktree."""
    env = os.environ.get("STS2_PCK")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "sts2_binaries/v0.107.1/SlayTheSpire2.pck"
        if candidate.is_file():
            return candidate
    return here.parents[2] / "sts2_binaries/v0.107.1/SlayTheSpire2.pck"


PCK = _resolve_pck()
COMMITTED = Path(__file__).resolve().parents[1] / "site" / "public" / "images" / "power_big"


@unittest.skipUnless(PCK.is_file(), f"vendored pck not present at {PCK}")
@unittest.skipUnless(COMMITTED.is_dir(), f"committed power_big not present at {COMMITTED}")
class CommittedBigIconGateTest(unittest.TestCase):
    """The committed `power_big/` tree against a fresh extraction.

    Same shape as `test_extract_vfx.CommittedArtifactGateTest`, and for the
    same reason: the tree is a generated artifact that ships to the art
    bucket, so a stale commit is invisible without something comparing it to
    the pck.
    """

    # Declared for the type checker, the same way `ExtractParticlesTest` does
    # — `setUpClass` assigns them.
    _tmp: tempfile.TemporaryDirectory[str]
    fresh: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        index = read_pck_index(PCK)
        with PCK.open("rb") as fh:

            def read(path: str) -> bytes | None:
                entry = index.get(path)
                if entry is None:
                    return None
                off, size = entry
                fh.seek(off)
                return fh.read(size)

            extract_power_big_icons(index, read, cls._tmp.name)
        cls.fresh = Path(cls._tmp.name) / "power_big"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_every_extracted_icon_is_committed_and_identical(self) -> None:
        fresh = {p.name: p.read_bytes() for p in self.fresh.iterdir()}
        committed = {p.name: p.read_bytes() for p in COMMITTED.iterdir()}
        self.assertEqual(
            sorted(fresh),
            sorted(committed),
            "power_big/ differs from a fresh extraction — re-run `just extract-images`",
        )
        differing = sorted(n for n, b in fresh.items() if committed.get(n) != b)
        self.assertEqual(differing, [], "stale power_big icons")

    def test_the_missing_power_fallback_icon_ships(self) -> None:
        # `PowerModel.ResolvedBigIconPath` falls back to `missing_power.png`
        # for the 14 powers that have a tray icon and no big one, so
        # `assets.power_big` needs it present to mirror that chain.
        self.assertIn("missing_power.png", {p.name for p in COMMITTED.iterdir()})


class TestOwnerlessSprites(unittest.TestCase):
    """`YummyCookie` has one sprite per character and none under its own id;
    the workbench asks for the id, so the ownerless default is copied there."""

    def test_copies_the_default_variant_to_the_model_id(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "yummy_cookie_ironclad.png").write_bytes(b"ironclad")
            self.assertEqual(write_ownerless_sprites("relic_atlas", d), 1)
            self.assertEqual((Path(d) / "yummy_cookie.png").read_bytes(), b"ironclad")
            # Idempotent, and never overwrites.
            self.assertEqual(write_ownerless_sprites("relic_atlas", d), 0)

    def test_other_atlases_are_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(write_ownerless_sprites("power_atlas", d), 0)
            self.assertEqual(os.listdir(d), [])


if __name__ == "__main__":
    unittest.main()
