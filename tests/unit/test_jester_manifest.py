import tempfile
import unittest
from pathlib import Path

from research_pipeline.cli.build_jester_manifest import build_jester_manifest
from research_pipeline.data.schema import validate_manifest_record

TRAIN_CSV = (
    "1;Swiping Left\n"
    "2;Zooming In With Two Fingers\n"
    "3;No gesture\n"
    "4;Turning Hand Clockwise\n"
    "5;Sliding Two Fingers Right\n"
)
VALIDATION_CSV = "10;Swiping Right\n"


class JesterManifestTests(unittest.TestCase):
    def _annotations_dir(self, tmp: str) -> Path:
        root = Path(tmp)
        (root / "jester-v1-train.csv").write_text(TRAIN_CSV, encoding="utf-8")
        (root / "jester-v1-validation.csv").write_text(VALIDATION_CSV, encoding="utf-8")
        return root

    def test_default_mapping_drops_non_target_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = build_jester_manifest(self._annotations_dir(tmp))
            by_id = {record.repetition_id: record.target_label for record in records}
            # Non-target (Turning Hand) and unused motion-equivalent (Sliding) drop out.
            self.assertEqual(
                by_id,
                {"1": "swipe_left", "2": "zoom_in", "3": "no_gesture", "10": "swipe_right"},
            )
            self.assertTrue(all(record.source_dataset == "jester" for record in records))
            splits = {record.repetition_id: record.split_group for record in records}
            self.assertEqual(splits["1"], "train")
            self.assertEqual(splits["10"], "test")  # validation split -> held-out test
            for record in records:
                validate_manifest_record(record)  # round-trips against the schema

    def test_options_expand_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = build_jester_manifest(
                self._annotations_dir(tmp),
                include_motion_equivalents=True,
                fold_non_target_as_no_gesture=True,
            )
            by_id = {record.repetition_id: record.target_label for record in records}
            self.assertEqual(by_id["5"], "swipe_right")  # sliding two fingers right
            self.assertEqual(by_id["4"], "no_gesture")  # turning hand -> hard negative

    def test_frame_folder_sets_span_and_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotations = self._annotations_dir(tmp)
            frames_root = Path(tmp) / "frames"
            clip = frames_root / "1"
            clip.mkdir(parents=True)
            for index in range(1, 7):  # 6 frames at 12 fps -> 500 ms
                (clip / f"{index:05d}.jpg").write_bytes(b"x")
            records = build_jester_manifest(annotations, frames_root)
            record = next(r for r in records if r.repetition_id == "1")
            self.assertTrue(record.raw_video_path.endswith("%05d.jpg"))
            self.assertEqual(record.clip_end_ms, 500)


if __name__ == "__main__":
    unittest.main()
