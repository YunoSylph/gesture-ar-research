import os
import tempfile
import unittest
from pathlib import Path

from research_pipeline.data.manifest import write_jsonl
from research_pipeline.data.schema import ManifestRecord
from research_pipeline.evaluation.domain import summarize_domain_manifests


class DomainReportTests(unittest.TestCase):
    def test_local_plan_reports_missing_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "local.jsonl"
            write_jsonl(
                manifest,
                [
                    ManifestRecord(
                        sample_id="local_1",
                        source_dataset="local_phone",
                        public_label="swipe_left",
                        target_label="swipe_left",
                        participant_id="p1",
                        session_id="s1",
                        repetition_id="1",
                        split_group="local",
                        raw_video_path=os.path.join("videos", "missing.mp4"),
                        extras={"capture_domain": "phone_rear_ar", "camera_view": "rear_world"},
                    )
                ],
            )
            report = summarize_domain_manifests([manifest])
            self.assertEqual(report["domain_transfer_status"], "local_plan_ready_waiting_for_videos")
            self.assertEqual(report["local_phone"]["missing_raw_video_count"], 1)
            self.assertEqual(report["by_capture_domain"]["phone_rear_ar"], 1)


if __name__ == "__main__":
    unittest.main()
