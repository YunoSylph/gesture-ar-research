import unittest

from research_pipeline.labels import remap_ipn_label, swap_mirrored_label


class LabelTests(unittest.TestCase):
    def test_ipn_mapping(self):
        self.assertEqual(remap_ipn_label(7), "swipe_left")
        self.assertEqual(remap_ipn_label("G05"), "swipe_left")
        self.assertEqual(remap_ipn_label("G06"), "swipe_right")
        self.assertEqual(remap_ipn_label("Th-right"), "swipe_right")
        self.assertEqual(remap_ipn_label("Zoom-o"), "zoom_out")

    def test_mirror_swap_rules(self):
        self.assertEqual(swap_mirrored_label("swipe_left"), "swipe_right")
        self.assertEqual(swap_mirrored_label("swipe_right"), "swipe_left")
        self.assertEqual(swap_mirrored_label("click_2f"), "click_2f")


if __name__ == "__main__":
    unittest.main()
