import json
import tempfile
import unittest
from pathlib import Path

from research_pipeline.cli.build_hagrid_manifest import build_hagrid_manifest
from research_pipeline.data.schema import validate_manifest_record


class HagridManifestTests(unittest.TestCase):
    def _annotations_dir(self, tmp: str) -> Path:
        root = Path(tmp)
        (root / "two_up.json").write_text(
            json.dumps(
                {
                    "00000000": {"user_id": "u1", "labels": ["two_up"]},
                    "00000001": {"user_id": "u2", "labels": ["two_up"]},
                }
            ),
            encoding="utf-8",
        )
        # A non-target gesture file that must be skipped by default.
        (root / "fist.json").write_text(
            json.dumps({"00000002": {"user_id": "u3", "labels": ["fist"]}}), encoding="utf-8"
        )
        return root

    def test_maps_two_finger_pose_and_tags_static(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = build_hagrid_manifest(self._annotations_dir(tmp), Path(tmp) / "imgs")
            self.assertEqual(len(records), 2)  # fist.json skipped
            for record in records:
                self.assertEqual(record.source_dataset, "hagrid")
                self.assertEqual(record.target_label, "point_2f")
                self.assertTrue(record.raw_video_path.endswith(".jpg"))
                self.assertEqual(record.extras.get("static_pose"), True)
                validate_manifest_record(record)
            self.assertEqual({r.participant_id for r in records}, {"u1", "u2"})

    def test_fold_non_target_includes_fist_as_no_gesture(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = build_hagrid_manifest(
                self._annotations_dir(tmp), fold_non_target_as_no_gesture=True
            )
            by_target = {}
            for record in records:
                by_target.setdefault(record.target_label, 0)
                by_target[record.target_label] += 1
            self.assertEqual(by_target, {"point_2f": 2, "no_gesture": 1})


if __name__ == "__main__":
    unittest.main()
