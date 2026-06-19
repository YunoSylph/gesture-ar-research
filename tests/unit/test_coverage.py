import unittest

import numpy as np

from research_pipeline.data.coverage import filter_records_by_coverage, tensor_coverage
from research_pipeline.data.schema import ManifestRecord
from research_pipeline.data.tensors import LandmarkTensor


def _record(sample_id: str, target: str) -> ManifestRecord:
    return ManifestRecord(
        sample_id=sample_id,
        source_dataset="local_phone",
        public_label=target,
        target_label=target,
        participant_id="p",
        session_id="s",
        repetition_id="1",
        split_group="local",
    )


def _tensor(mask_values: list[bool]) -> LandmarkTensor:
    t = len(mask_values)
    mask = np.array(mask_values, dtype=bool)
    return LandmarkTensor(
        landmarks=np.zeros((t, 21, 3), dtype=np.float32),
        sequence_mask=mask,
        frame_confidence=mask.astype(np.float32),
        handedness_score=mask.astype(np.float32),
    )


class CoverageTests(unittest.TestCase):
    def test_tensor_coverage(self):
        self.assertEqual(tensor_coverage(_tensor([True, True, True, True])), 1.0)
        self.assertEqual(tensor_coverage(_tensor([True, False, True, False])), 0.5)
        self.assertEqual(tensor_coverage(_tensor([False, False])), 0.0)

    def test_filter_drops_low_coverage(self):
        records = [_record("a", "click_2f"), _record("b", "swipe_right"), _record("c", "swipe_right")]
        coverage = {"a": 0.98, "b": 0.50, "c": 0.90}
        kept, report = filter_records_by_coverage(
            records, lambda r: coverage[r.sample_id], min_coverage=0.85
        )
        self.assertEqual([r.sample_id for r in kept], ["a", "c"])
        self.assertEqual(report.kept, 2)
        self.assertEqual(report.dropped, 1)
        self.assertEqual(report.dropped_by_target, {"swipe_right": 1})
        self.assertEqual(report.kept_by_target, {"click_2f": 1, "swipe_right": 1})
        self.assertAlmostEqual(report.mean_coverage, (0.98 + 0.50 + 0.90) / 3)


if __name__ == "__main__":
    unittest.main()
