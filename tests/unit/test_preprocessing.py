import unittest

from research_pipeline.data.synthetic import synthetic_landmarks
from research_pipeline.features.augment import mirrored_target_label, mirror_landmarks
from research_pipeline.features.preprocessing import preprocess_dual_view


class PreprocessingTests(unittest.TestCase):
    def test_motion_stream_preserves_swipe_direction(self):
        left = preprocess_dual_view(synthetic_landmarks("swipe_left", seed=1), target_length=32)
        right = preprocess_dual_view(synthetic_landmarks("swipe_right", seed=1), target_length=32)
        wrist_x_start = 2
        self.assertLess(left.motion[-1, wrist_x_start] - left.motion[0, wrist_x_start], 0)
        self.assertGreater(right.motion[-1, wrist_x_start] - right.motion[0, wrist_x_start], 0)

    def test_mirror_changes_directed_label(self):
        tensor = synthetic_landmarks("swipe_left", seed=2)
        mirrored = mirror_landmarks(tensor)
        self.assertAlmostEqual(float(mirrored.landmarks[0, 0, 0]), 1.0 - float(tensor.landmarks[0, 0, 0]), places=5)
        self.assertEqual(mirrored_target_label("swipe_left"), "swipe_right")


if __name__ == "__main__":
    unittest.main()

