import unittest

from research_pipeline.data.merge import merge_manifests
from research_pipeline.data.schema import ManifestRecord
from research_pipeline.utils.errors import SchemaError


def _record(sample_id: str, target: str, source: str, split: str = "train") -> ManifestRecord:
    return ManifestRecord(
        sample_id=sample_id,
        source_dataset=source,
        public_label=target,
        target_label=target,
        participant_id="p",
        session_id="s",
        repetition_id="1",
        split_group=split,
    )


def _corpus() -> list[ManifestRecord]:
    records = []
    records += [_record(f"ng{i}", "no_gesture", "jester") for i in range(10)]
    records += [_record(f"sl{i}", "swipe_left", "jester") for i in range(10)]
    records += [_record(f"pt{i}", "point_2f", "local_phone") for i in range(2)]
    return records


class MergeTests(unittest.TestCase):
    def test_per_source_cap_balances_dominant_source(self):
        merged, report = merge_manifests(_corpus(), max_per_class_per_source=3, seed=7)
        self.assertEqual(report.by_target, {"no_gesture": 3, "swipe_left": 3, "point_2f": 2})
        self.assertEqual(report.total, 8)
        self.assertEqual(report.dropped, 14)  # 7 dropped from each capped jester class
        # domain mix is preserved in the report
        self.assertEqual(report.by_target_source["point_2f"], {"local_phone": 2})

    def test_missing_targets_reported(self):
        _merged, report = merge_manifests(_corpus())
        self.assertIn("click_2f", report.missing_targets)
        self.assertIn("zoom_in", report.missing_targets)
        self.assertNotIn("no_gesture", report.missing_targets)

    def test_subsampling_is_deterministic_and_order_preserving(self):
        first, _ = merge_manifests(_corpus(), max_per_class_per_source=3, seed=7)
        second, _ = merge_manifests(_corpus(), max_per_class_per_source=3, seed=7)
        # same seed -> identical kept set in identical order
        self.assertEqual([r.sample_id for r in first], [r.sample_id for r in second])
        # output preserves the input concatenation order (kept indices ascending)
        original_index = {record.sample_id: index for index, record in enumerate(_corpus())}
        kept_indices = [original_index[r.sample_id] for r in first]
        self.assertEqual(kept_indices, sorted(kept_indices))

    def test_duplicate_sample_ids_rejected(self):
        records = [_record("dup", "no_gesture", "jester"), _record("dup", "swipe_left", "jester")]
        with self.assertRaises(SchemaError):
            merge_manifests(records)


if __name__ == "__main__":
    unittest.main()
