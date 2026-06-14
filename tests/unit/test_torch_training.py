import os
import tempfile
import unittest
from pathlib import Path

from research_pipeline.data.manifest import write_jsonl
from research_pipeline.data.schema import ManifestRecord
from research_pipeline.data.synthetic import synthetic_landmarks
from research_pipeline.data.tensors import save_landmark_npz
from research_pipeline.labels import TARGET_LABELS
from research_pipeline.models.torch_training import train_tcn
from research_pipeline.utils.errors import DependencyMissingError


class TorchTrainingTests(unittest.TestCase):
    def test_tcn_records_validation_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            tensor_dir = root / "landmarks"
            records = []
            for label in TARGET_LABELS:
                for repetition in range(3):
                    sample_id = f"{label}_{repetition}"
                    tensor_path = tensor_dir / f"{sample_id}.npz"
                    save_landmark_npz(
                        tensor_path,
                        synthetic_landmarks(label, length=16, seed=7, sample_id=sample_id),
                    )
                    records.append(
                        ManifestRecord(
                            sample_id=sample_id,
                            source_dataset="synthetic",
                            public_label=label,
                            target_label=label,
                            participant_id=f"p{repetition}",
                            session_id="unit",
                            repetition_id=str(repetition),
                            split_group="train",
                            tensor_path=os.path.relpath(tensor_path, manifest.parent),
                        )
                    )
            write_jsonl(manifest, records)
            try:
                artifact = train_tcn(
                    manifest,
                    root / "model.pkl",
                    target_length=16,
                    epochs=2,
                    batch_size=8,
                    validation_split=0.33,
                    early_stopping_patience=1,
                )
            except DependencyMissingError:
                self.skipTest("PyTorch is not installed")
            self.assertEqual(artifact["training"]["validation_source"], "stratified_split")
            self.assertGreater(artifact["training"]["validation_samples"], 0)
            self.assertIn("validation_loss", artifact["history"][0])


if __name__ == "__main__":
    unittest.main()
