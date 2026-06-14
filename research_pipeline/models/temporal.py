from __future__ import annotations

from pathlib import Path

import numpy as np

from research_pipeline.data.manifest import read_jsonl
from research_pipeline.data.schema import resolve_path
from research_pipeline.data.tensors import load_landmark_npz
from research_pipeline.features.preprocessing import preprocess_dual_view
from research_pipeline.labels import TARGET_LABELS, label_to_index
from research_pipeline.models.artifacts import save_artifact
from research_pipeline.utils.errors import SchemaError
from research_pipeline.utils.random import set_global_seed


def train_temporal_prototype(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 13,
    target_length: int = 32,
) -> dict:
    """Small dependency-free temporal baseline used for smoke and ablation sanity checks."""

    set_global_seed(seed)
    records = read_jsonl(manifest_path)
    base_dir = Path(manifest_path).parent
    vectors_by_label: dict[int, list[np.ndarray]] = {}
    for record in records:
        if not record.tensor_path:
            raise SchemaError(f"Record '{record.sample_id}' has no tensor_path.")
        tensor = load_landmark_npz(resolve_path(record.tensor_path, base_dir))
        sequence = preprocess_dual_view(tensor, target_length=target_length)
        label_index = label_to_index(record.target_label)
        vectors_by_label.setdefault(label_index, []).append(sequence.features.reshape(-1))
    if not vectors_by_label:
        raise SchemaError("Cannot train temporal prototype on an empty manifest.")
    centroids = {
        index: np.vstack(vectors).mean(axis=0).astype(np.float32)
        for index, vectors in vectors_by_label.items()
    }
    artifact = {
        "model_type": "temporal_prototype",
        "labels": list(TARGET_LABELS),
        "target_length": target_length,
        "centroids": centroids,
        "seed": seed,
    }
    save_artifact(output_path, artifact)
    return artifact

