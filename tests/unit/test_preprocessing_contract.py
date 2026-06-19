import unittest

import numpy as np

from research_pipeline.data.tensors import LandmarkTensor
from research_pipeline.features.preprocessing import preprocess_dual_view
from research_pipeline.models.preprocessing_contract import feature_layout_contract, golden_sample


def _tensor(length: int = 20) -> LandmarkTensor:
    rng = np.random.default_rng(0)
    return LandmarkTensor(
        landmarks=rng.random((length, 21, 3)).astype(np.float32),
        sequence_mask=np.ones((length,), dtype=bool),
        frame_confidence=np.ones((length,), dtype=np.float32),
        handedness_score=np.ones((length,), dtype=np.float32),
    )


class PreprocessingContractTests(unittest.TestCase):
    def test_contract_matches_real_feature_dim(self):
        contract = feature_layout_contract(target_length=32, multiview_coords=2)
        self.assertEqual(contract["feature_dim"], 326)
        features = preprocess_dual_view(
            _tensor(), target_length=32, include_multiview=True, multiview_coords=2
        ).features
        self.assertEqual(features.shape, (32, contract["feature_dim"]))

    def test_non_multiview_contract_is_dual_view_74(self):
        contract = feature_layout_contract(include_multiview=False)
        self.assertEqual(contract["feature_dim"], 74)
        self.assertEqual([b["name"] for b in contract["blocks"]], ["pose", "motion"])
        features = preprocess_dual_view(
            _tensor(), target_length=32, include_multiview=False
        ).features
        self.assertEqual(features.shape, (32, contract["feature_dim"]))

    def test_blocks_are_contiguous_and_sum_to_dim(self):
        contract = feature_layout_contract()
        cursor = 0
        for block in contract["blocks"]:
            self.assertEqual(block["start"], cursor)
            self.assertEqual(block["end"] - block["start"], block["dim"])
            cursor = block["end"]
        self.assertEqual(cursor, contract["feature_dim"])
        names = [b["name"] for b in contract["blocks"]]
        self.assertEqual(names, ["pose", "motion", "jcd", "slow_motion", "fast_motion"])

    def test_golden_sample_is_self_consistent(self):
        tensor = _tensor()
        sample = golden_sample(tensor, sample_id="s", target_label="click_2f")
        expected = np.asarray(sample["expected_features"], dtype=np.float32)
        recomputed = preprocess_dual_view(
            tensor, target_length=32, include_multiview=True, multiview_coords=2
        ).features
        self.assertEqual(expected.shape, (32, 326))
        self.assertTrue(np.allclose(expected, recomputed, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
