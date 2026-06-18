from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from research_pipeline.data.tensors import LandmarkTensor
from research_pipeline.features.preprocessing import clip_feature_summary, preprocess_dual_view
from research_pipeline.labels import TARGET_LABELS, label_to_index
from research_pipeline.models.common import Prediction, prediction_from_scores
from research_pipeline.models.rule_based import RuleBasedRecognizer
from research_pipeline.utils.errors import PipelineError


def save_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(artifact, handle)


def load_artifact(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict) or "model_type" not in artifact:
        raise PipelineError(f"'{path}' is not a research_pipeline model artifact.")
    return artifact


def artifact_feature_flags(artifact: dict[str, Any]) -> tuple[bool, int]:
    """Read the feature-set flags an artifact was trained with (default dual-view)."""

    features_cfg = artifact.get("features", {})
    return bool(features_cfg.get("include_multiview", False)), int(features_cfg.get("multiview_coords", 2))


def predict_with_artifact(artifact: dict[str, Any], tensor: LandmarkTensor) -> Prediction:
    model_type = artifact["model_type"]
    target_length = int(artifact.get("target_length", 32))
    include_multiview, multiview_coords = artifact_feature_flags(artifact)
    sequence = preprocess_dual_view(
        tensor,
        target_length=target_length,
        include_multiview=include_multiview,
        multiview_coords=multiview_coords,
    )

    if model_type == "c0_rule":
        recognizer = RuleBasedRecognizer(**artifact.get("params", {}))
        return recognizer.predict(tensor)

    if model_type == "c1_random_forest":
        estimator = artifact["estimator"]
        summary = clip_feature_summary(sequence)[None, :]
        labels = artifact["labels"]
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(summary)[0]
            scores = {label: 0.0 for label in TARGET_LABELS}
            for cls, score in zip(estimator.classes_, proba):
                scores[labels[int(cls)]] = float(score)
            return prediction_from_scores(scores)
        cls = int(estimator.predict(summary)[0])
        label = labels[cls]
        return prediction_from_scores({label: 1.0})

    if model_type == "temporal_prototype":
        centroids = artifact["centroids"]
        labels = artifact["labels"]
        vector = sequence.features.reshape(-1)
        distances = {
            labels[index]: float(np.linalg.norm(vector - centroid))
            for index, centroid in centroids.items()
        }
        scores = {label: 1.0 / (1.0 + distances.get(label, 1e6)) for label in TARGET_LABELS}
        return prediction_from_scores(scores)

    if model_type == "c1t_tcn_torch":
        import torch

        from research_pipeline.models.tcn import TCNConfig, build_tcn

        config = TCNConfig(**artifact["tcn_config"])
        model = build_tcn(config)
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(sequence.features[None, :, :].astype(np.float32))
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        labels = artifact["labels"]
        scores = {label: 0.0 for label in TARGET_LABELS}
        for index, score in enumerate(probabilities):
            scores[labels[index]] = float(score)
        return prediction_from_scores(scores)

    raise PipelineError(f"Unsupported model_type '{model_type}'.")


def labels_to_indices(labels: list[str]) -> np.ndarray:
    return np.array([label_to_index(label) for label in labels], dtype=np.int64)
