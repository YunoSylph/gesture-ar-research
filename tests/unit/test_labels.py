import unittest

from research_pipeline.labels import (
    HAGRID_MISSING_TARGETS,
    JESTER_MISSING_TARGETS,
    TARGET_LABELS,
    remap_hagrid_label,
    remap_ipn_label,
    remap_jester_label,
    swap_mirrored_label,
)


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

    def test_jester_core_mapping(self):
        self.assertEqual(remap_jester_label("Swiping Left"), "swipe_left")
        self.assertEqual(remap_jester_label("Swiping Right"), "swipe_right")
        self.assertEqual(remap_jester_label("Zooming In With Two Fingers"), "zoom_in")
        self.assertEqual(remap_jester_label("Zooming Out With Two Fingers"), "zoom_out")
        self.assertEqual(remap_jester_label("No gesture"), "no_gesture")
        self.assertEqual(remap_jester_label("Doing other things"), "no_gesture")
        # tolerant of casing / whitespace noise
        self.assertEqual(remap_jester_label("  swiping   left "), "swipe_left")
        for name in ("Swiping Left", "Zooming In With Two Fingers", "No gesture"):
            self.assertIn(remap_jester_label(name), TARGET_LABELS)

    def test_jester_vocabulary_gap(self):
        # point_2f / click_2f have no Jester source and must come from elsewhere.
        self.assertIn("point_2f", JESTER_MISSING_TARGETS)
        self.assertIn("click_2f", JESTER_MISSING_TARGETS)

    def test_jester_non_target_and_options(self):
        # Non-command motion is excluded by default, foldable as a hard negative.
        self.assertIsNone(remap_jester_label("Turning Hand Clockwise"))
        self.assertEqual(
            remap_jester_label("Turning Hand Clockwise", fold_non_target_as_no_gesture=True),
            "no_gesture",
        )
        # Motion-equivalents are off by default, mapped only when requested.
        self.assertIsNone(remap_jester_label("Sliding Two Fingers Left"))
        self.assertEqual(
            remap_jester_label("Sliding Two Fingers Left", include_motion_equivalents=True),
            "swipe_left",
        )
        self.assertEqual(
            remap_jester_label("Zooming In With Full Hand", include_motion_equivalents=True),
            "zoom_in",
        )
        # Unknown / empty labels are ignored.
        self.assertIsNone(remap_jester_label("definitely not a jester class"))
        self.assertIsNone(remap_jester_label(""))

    def test_hagrid_core_mapping(self):
        # Two-finger poses -> point_2f; HaGRID's single-finger "point" must not map.
        self.assertEqual(remap_hagrid_label("two_up"), "point_2f")
        self.assertEqual(remap_hagrid_label("two_up_inverted"), "point_2f")
        self.assertEqual(remap_hagrid_label("peace"), "point_2f")
        self.assertEqual(remap_hagrid_label("peace_inverted"), "point_2f")
        self.assertEqual(remap_hagrid_label("no_gesture"), "no_gesture")
        self.assertIsNone(remap_hagrid_label("point"))  # single-finger point
        self.assertIsNone(remap_hagrid_label("one"))

    def test_hagrid_gaps_and_fold(self):
        # Static HaGRID cannot supply the dynamic classes or click_2f.
        for label in ("swipe_left", "swipe_right", "zoom_in", "zoom_out", "click_2f"):
            self.assertIn(label, HAGRID_MISSING_TARGETS)
        # Non-target poses excluded by default, foldable as static hard negatives.
        self.assertIsNone(remap_hagrid_label("fist"))
        self.assertEqual(
            remap_hagrid_label("fist", fold_non_target_as_no_gesture=True), "no_gesture"
        )
        # Folding only applies to genuine HaGRID classes, not arbitrary strings.
        self.assertIsNone(
            remap_hagrid_label("not_a_hagrid_class", fold_non_target_as_no_gesture=True)
        )


if __name__ == "__main__":
    unittest.main()
