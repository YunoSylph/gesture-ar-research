import unittest

import numpy as np

from research_pipeline.cli.extract_landmarks import _is_static_image, _replicate_frame_to_clip


class StaticExtractionTests(unittest.TestCase):
    def test_is_static_image_distinguishes_sources(self):
        self.assertTrue(_is_static_image("data/hagrid/two_up/00000000.jpg"))
        self.assertTrue(_is_static_image("img.PNG"))
        # Jester image-sequence pattern and real videos are not static stills.
        self.assertFalse(_is_static_image("data/jester/1/%05d.jpg"))
        self.assertFalse(_is_static_image("clip.avi"))
        self.assertFalse(_is_static_image(""))

    def test_replicate_frame_builds_uniform_clip(self):
        frame = np.random.default_rng(0).random((21, 3)).astype(np.float32)
        tensor = _replicate_frame_to_clip(frame, detected=True, confidence=0.9, target_length=32)
        self.assertEqual(tensor.landmarks.shape, (32, 21, 3))
        # every frame is the same replicated pose
        self.assertTrue(np.allclose(tensor.landmarks[0], tensor.landmarks[-1]))
        self.assertTrue(np.allclose(tensor.landmarks, frame[None, :, :]))
        self.assertTrue(tensor.sequence_mask.all())
        self.assertTrue(np.allclose(tensor.frame_confidence, 0.9))

    def test_replicate_frame_undetected_is_masked(self):
        tensor = _replicate_frame_to_clip(
            np.zeros((21, 3), dtype=np.float32), detected=False, confidence=0.0, target_length=16
        )
        self.assertEqual(tensor.landmarks.shape, (16, 21, 3))
        self.assertFalse(tensor.sequence_mask.any())
        self.assertTrue(np.allclose(tensor.frame_confidence, 0.0))


if __name__ == "__main__":
    unittest.main()
