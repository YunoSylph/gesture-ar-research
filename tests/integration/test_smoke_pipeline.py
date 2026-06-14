import os
import tempfile
import unittest
from pathlib import Path

from research_pipeline.data.manifest import write_jsonl
from research_pipeline.data.schema import ManifestRecord
from research_pipeline.data.synthetic import synthetic_landmarks
from research_pipeline.data.tensors import save_landmark_npz
from research_pipeline.evaluation.recognition import benchmark_recognition_manifest
from research_pipeline.labels import TARGET_LABELS
from research_pipeline.models.temporal import train_temporal_prototype


class SmokePipelineTests(unittest.TestCase):
    def test_synthetic_train_and_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            tensor_dir = root / "landmarks"
            records = []
            for label in TARGET_LABELS:
                sample_id = f"{label}_0"
                tensor_path = tensor_dir / f"{sample_id}.npz"
                save_landmark_npz(tensor_path, synthetic_landmarks(label, seed=42, sample_id=sample_id))
                records.append(
                    ManifestRecord(
                        sample_id=sample_id,
                        source_dataset="synthetic",
                        public_label=label,
                        target_label=label,
                        participant_id="p1",
                        session_id="s1",
                        repetition_id="0",
                        split_group="smoke",
                        tensor_path=os.path.relpath(tensor_path, manifest.parent),
                    )
                )
            write_jsonl(manifest, records)
            model_path = root / "model.pkl"
            train_temporal_prototype(manifest, model_path)
            metrics, latency = benchmark_recognition_manifest(manifest, model_path=model_path)
            self.assertGreaterEqual(metrics.accuracy, 0.85)
            self.assertEqual(latency["num_samples"], len(TARGET_LABELS))


if __name__ == "__main__":
    unittest.main()

