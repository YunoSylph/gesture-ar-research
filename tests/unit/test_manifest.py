import tempfile
import unittest
from pathlib import Path

from research_pipeline.data.manifest import read_jsonl, write_jsonl
from research_pipeline.data.schema import ManifestRecord
from research_pipeline.cli.ingest_local_videos import build_local_manifest_records


class ManifestTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            record = ManifestRecord(
                sample_id="s1",
                source_dataset="synthetic",
                public_label="swipe_left",
                target_label="swipe_left",
                participant_id="p1",
                session_id="session",
                repetition_id="1",
                split_group="train",
            )
            write_jsonl(path, [record])
            loaded = read_jsonl(path)
            self.assertEqual(loaded[0].sample_id, "s1")
            self.assertEqual(loaded[0].target_label, "swipe_left")

    def test_local_ingest_preserves_domain_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "capture.csv"
            csv_path.write_text(
                "file_name,target_label,capture_domain,camera_view,coordinate_semantics\n"
                "clip.mp4,swipe_left,phone_rear_ar,rear_world,screen_space\n",
                encoding="utf-8",
            )
            records = build_local_manifest_records(csv_path, root / "videos")
            self.assertEqual(records[0].source_dataset, "local_phone")
            self.assertEqual(records[0].extras["capture_domain"], "phone_rear_ar")
            self.assertEqual(records[0].extras["camera_view"], "rear_world")
            self.assertEqual(records[0].extras["coordinate_semantics"], "screen_space")


if __name__ == "__main__":
    unittest.main()
