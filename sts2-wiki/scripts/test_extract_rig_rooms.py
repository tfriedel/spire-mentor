"""Path rules in `extract_rig` (atlas page sizes) and `extract_rooms` (outputs)."""

import unittest

from scripts.extract_rig import _page_sizes
from scripts.extract_rooms import aliases_for, output_for


class TestAtlasPageSizes(unittest.TestCase):
    """The atlas's own page sizes pick the right texture when an unrelated
    imported texture shares the page's name (battleworn_dummy.png is both the
    396x502 rig page and the 3440x1613 event illustration)."""

    def test_reads_each_page_size(self):
        atlas = [
            "battleworn_dummy.png",
            "size:396,502",
            "filter:Linear,Linear",
            "arm",
            "bounds:1,2,3,4",
            "",
            "battleworn_dummy_2.png",
            "size: 80, 186",
            "leg",
        ]
        self.assertEqual(
            _page_sizes(atlas),
            {"battleworn_dummy.png": (396, 502), "battleworn_dummy_2.png": (80, 186)},
        )

    def test_region_sizes_are_not_page_sizes(self):
        atlas = ["a.png", "filter:Linear,Linear", "region", "size:9,9"]
        self.assertEqual(_page_sizes(atlas), {})


class TestRoomOutputs(unittest.TestCase):
    def test_ancient_portraits_are_the_map_node_icons(self):
        rel = output_for("images/packed/map/ancients/ancient_node_neow.png")
        self.assertEqual(rel, "ui/map/ancients/ancient_node_neow.png")
        self.assertEqual(aliases_for(rel), ["ancients/neow.png"])
        self.assertEqual(aliases_for("ui/map/ancients/ancient_node_neow_outline.png"), [])

    def test_ancient_scene_backdrops_are_not_portraits(self):
        self.assertIsNone(output_for("images/ancients/darv_placeholder.png"))

    def test_static_monster_sprites(self):
        self.assertEqual(
            output_for("images/monsters/hourglass_placeholder.png"),
            "monsters/hourglass_placeholder.png",
        )
        self.assertIsNone(output_for("images/monsters/beta/the_spear.png"))


if __name__ == "__main__":
    unittest.main()
