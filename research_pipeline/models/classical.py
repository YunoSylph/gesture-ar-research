from __future__ import annotations

from pathlib import Path

import numpy as np

from research_pipeline.data.manifest import read_jsonl
from research_pipeline.data.schema import resolve_path
from research_pipeline.data.tensors import load_landmark_npz
from research_pipeline.features.preprocessing import clip_feature_summary, preprocess_dual_view
from research_pipeline.labels import TARGET_LABELS, label_to_index
from research_pipeline.models.artifacts import save_artifact
from research_pipeline.utils.errors import DependencyMissingError, SchemaError
from research_pipeline.utils.random import set_global_seed


def train_random_forest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 13,
    target_length: int = 32,
    n_estimators: int = 200,
) -> dict:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise DependencyMissingError("scikit-learn is required for C1 random forest training.") from exc

    set_global_seed(seed)
    records = read_jsonl(manifest_path)
    base_dir = Path(manifest_path).parent
    features: list[np.ndarray] = []
    labels: list[int] = []
    for record in records:
        if not record.tensor_path:
            raise SchemaError(f"Record '{record.sample_id}' has no tensor_path.")
        tensor = load_landmark_npz(resolve_path(record.tensor_path, base_dir))
        sequence = preprocess_dual_view(tensor, target_length=target_length)
        features.append(clip_feature_summary(sequence))
        labels.append(label_to_index(record.target_label))
    if not features:
        raise SchemaError("Cannot train C1 model on an empty manifest.")

    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    estimator.fit(np.vstack(features), np.array(labels, dtype=np.int64))
    artifact = {
        "model_type": "c1_random_forest",
        "labels": list(TARGET_LABELS),
        "target_length": target_length,
        "estimator": estimator,
        "seed": seed,
    }
    save_artifact(output_path, artifact)
    return artifact

