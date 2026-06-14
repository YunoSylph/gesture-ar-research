from __future__ import annotations

import os
from pathlib import Path

from research_pipeline.data.manifest import write_jsonl
from research_pipeline.data.schema import ManifestRecord
from research_pipeline.data.synthetic import synthetic_landmarks
from research_pipeline.data.tensors import save_landmark_npz
from research_pipeline.evaluation.recognition import benchmark_recognition_manifest
from research_pipeline.labels import TARGET_LABELS
from research_pipeline.models.temporal import train_temporal_prototype


def main() -> None:
    root = Path("artifacts/smoke/public")
    manifest_path = root / "synthetic_landmarks.jsonl"
    tensor_dir = root / "landmarks"
    model_path = root / "temporal_prototype.pkl"
    records: list[ManifestRecord] = []
    for label in TARGET_LABELS:
        for repetition in range(4):
            sample_id = f"smoke_{label}_{repetition:02d}"
            tensor_path = tensor_dir / f"{sample_id}.npz"
            tensor = synthetic_landmarks(label, length=32, seed=101, sample_id=sample_id)
            save_landmark_npz(tensor_path, tensor, sample_id=sample_id, target_label=label)
            records.append(
                ManifestRecord(
                    sample_id=sample_id,
                    source_dataset="synthetic",
                    public_label=label,
                    target_label=label,
                    participant_id=f"p{repetition % 2}",
                    session_id="smoke",
                    repetition_id=str(repetition),
                    split_group="smoke",
                    tensor_path=os.path.relpath(tensor_path, manifest_path.parent),
                    notes="generated_by_smoke_public",
                )
            )
    write_jsonl(manifest_path, records)
    train_temporal_prototype(manifest_path, model_path, seed=101, target_length=32)
    metrics, latency = benchmark_recognition_manifest(manifest_path, model_path=model_path)
    print(f"smoke_public ok: samples={len(records)} accuracy={metrics.accuracy:.3f} p95_ms={latency['offline_latency_ms_p95']:.3f}")


if __name__ == "__main__":
    main()

