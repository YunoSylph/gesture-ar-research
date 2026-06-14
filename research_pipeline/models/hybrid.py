from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from research_pipeline.data.tensors import LandmarkTensor
from research_pipeline.features.preprocessing import clip_feature_summary, palm_scale, preprocess_dual_view
from research_pipeline.labels import TARGET_LABELS
from research_pipeline.models.common import Prediction, prediction_from_scores
from research_pipeline.models.rule_based import RuleBasedRecognizer
from research_pipeline.utils.errors import PipelineError


@dataclass(slots=True)
class HybridConfig:
    neural_weight: float = 0.94
    geometry_weight: float = 0.06
    enable_safety_gate: bool = True
    action_threshold: float = 0.46
    no_gesture_margin: float = 0.03
    low_quality_confidence: float = 0.42
    low_quality_no_gesture_boost: float = 0.12
    weak_motion_no_gesture_boost: float = 0.06
    swipe_motion_min: float = 0.055
    zoom_delta_min: float = 0.035
    click_distance_max: float = 0.075


class CachedArtifactPredictor:
    """Inference wrapper that keeps heavyweight model objects alive across samples."""

    def __init__(self, artifact: dict[str, Any]):
        self.artifact = artifact
        self.model_type = artifact["model_type"]
        self.target_length = int(artifact.get("target_length", 32))
        self._torch_model = None

        if self.model_type == "c1t_tcn_torch":
            import torch

            from research_pipeline.models.tcn import TCNConfig, build_tcn

            config = TCNConfig(**artifact["tcn_config"])
            model = build_tcn(config)
            model.load_state_dict(artifact["state_dict"])
            model.eval()
            self._torch = torch
            self._torch_model = model
        elif self.model_type not in {"c0_rule", "c1_random_forest", "temporal_prototype"}:
            raise PipelineError(f"Unsupported model_type '{self.model_type}'.")

    def predict(self, tensor: LandmarkTensor) -> Prediction:
        sequence = preprocess_dual_view(tensor, target_length=self.target_length)

        if self.model_type == "c0_rule":
            recognizer = RuleBasedRecognizer(**self.artifact.get("params", {}))
            return recognizer.predict(tensor)

        if self.model_type == "c1_random_forest":
            estimator = self.artifact["estimator"]
            summary = clip_feature_summary(sequence)[None, :]
            labels = self.artifact["labels"]
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(summary)[0]
                scores = {label: 0.0 for label in TARGET_LABELS}
                for cls, score in zip(estimator.classes_, proba):
                    scores[labels[int(cls)]] = float(score)
                return prediction_from_scores(scores)
            cls = int(estimator.predict(summary)[0])
            return prediction_from_scores({labels[cls]: 1.0})

        if self.model_type == "temporal_prototype":
            centroids = self.artifact["centroids"]
            labels = self.artifact["labels"]
            vector = sequence.features.reshape(-1)
            distances = {
                labels[index]: float(np.linalg.norm(vector - centroid))
                for index, centroid in centroids.items()
            }
            scores = {label: 1.0 / (1.0 + distances.get(label, 1e6)) for label in TARGET_LABELS}
            return prediction_from_scores(scores)

        if self.model_type == "c1t_tcn_torch":
            assert self._torch_model is not None
            with self._torch.no_grad():
                x = self._torch.from_numpy(sequence.features[None, :, :].astype(np.float32))
                logits = self._torch_model(x)
                probabilities = self._torch.softmax(logits, dim=1)[0].cpu().numpy()
            labels = self.artifact["labels"]
            scores = {label: 0.0 for label in TARGET_LABELS}
            for index, score in enumerate(probabilities):
                scores[labels[index]] = float(score)
            return prediction_from_scores(scores)

        raise PipelineError(f"Unsupported model_type '{self.model_type}'.")


class HybridRecognizer:
    def __init__(self, artifact: dict[str, Any], config: HybridConfig | None = None):
        self.neural = CachedArtifactPredictor(artifact)
        self.rule = RuleBasedRecognizer()
        self.config = config or HybridConfig()

    def predict(self, tensor: LandmarkTensor) -> Prediction:
        neural = self.neural.predict(tensor)
        geometry = self._geometry_prior(tensor)
        return fuse_hybrid_predictions(neural, geometry, tensor, self.config)

    def _geometry_prior(self, tensor: LandmarkTensor) -> Prediction:
        return geometry_prior_prediction(tensor, self.config, self.rule)

    def _apply_safety_gate(self, scores: dict[str, float], tensor: LandmarkTensor) -> dict[str, float]:
        return apply_hybrid_safety_gate(scores, tensor, self.config)


class GeometryPriorRecognizer:
    def __init__(self, config: HybridConfig | None = None):
        self.rule = RuleBasedRecognizer()
        self.config = config or HybridConfig()

    def predict(self, tensor: LandmarkTensor) -> Prediction:
        return geometry_prior_prediction(tensor, self.config, self.rule)


def geometry_prior_prediction(
    tensor: LandmarkTensor,
    config: HybridConfig | None = None,
    rule: RuleBasedRecognizer | None = None,
) -> Prediction:
    config = config or HybridConfig()
    recognizer = rule or RuleBasedRecognizer()
    rule_prediction = recognizer.predict(tensor)
    stats = _geometry_stats(tensor)
    scores = dict(rule_prediction.scores)

    if abs(stats["dx"]) < config.swipe_motion_min:
        scores["swipe_left"] *= 0.45
        scores["swipe_right"] *= 0.45
    if abs(stats["scale_delta"]) < config.zoom_delta_min:
        scores["zoom_in"] *= 0.55
        scores["zoom_out"] *= 0.55
    if stats["index_middle_min"] > config.click_distance_max:
        scores["click_2f"] *= 0.45
    if stats["motion"] < config.swipe_motion_min and abs(stats["scale_delta"]) < config.zoom_delta_min:
        scores["no_gesture"] += config.weak_motion_no_gesture_boost
    if stats["confidence"] < config.low_quality_confidence:
        scores["no_gesture"] += config.low_quality_no_gesture_boost
    return prediction_from_scores(scores)


def fuse_hybrid_predictions(
    neural: Prediction,
    geometry: Prediction,
    tensor: LandmarkTensor,
    config: HybridConfig | None = None,
) -> Prediction:
    config = config or HybridConfig()
    scores = {
        label: config.neural_weight * neural.scores.get(label, 0.0)
        + config.geometry_weight * geometry.scores.get(label, 0.0)
        for label in TARGET_LABELS
    }
    if config.enable_safety_gate:
        scores = apply_hybrid_safety_gate(scores, tensor, config)
    return prediction_from_scores(scores)


def apply_hybrid_safety_gate(
    scores: dict[str, float],
    tensor: LandmarkTensor,
    config: HybridConfig | None = None,
) -> dict[str, float]:
    config = config or HybridConfig()
    scores = dict(scores)
    stats = _geometry_stats(tensor)
    top_label = max(scores, key=scores.get)
    if top_label == "no_gesture":
        return scores

    top_score = float(scores[top_label])
    no_score = float(scores.get("no_gesture", 0.0))
    if top_score < config.action_threshold and no_score + config.no_gesture_margin >= top_score:
        scores["no_gesture"] = max(scores["no_gesture"], top_score + config.no_gesture_margin)

    if stats["confidence"] < config.low_quality_confidence:
        scores["no_gesture"] += config.low_quality_no_gesture_boost

    if top_label in {"swipe_left", "swipe_right"} and abs(stats["dx"]) < config.swipe_motion_min:
        scores["no_gesture"] += config.weak_motion_no_gesture_boost
    if top_label in {"zoom_in", "zoom_out"} and abs(stats["scale_delta"]) < config.zoom_delta_min:
        scores["no_gesture"] += config.weak_motion_no_gesture_boost
    if top_label == "click_2f" and stats["index_middle_min"] > config.click_distance_max:
        scores["no_gesture"] += config.weak_motion_no_gesture_boost
    return scores


def _geometry_stats(tensor: LandmarkTensor) -> dict[str, float]:
    valid = tensor.sequence_mask.astype(bool)
    landmarks = tensor.landmarks[valid] if valid.shape[0] == tensor.landmarks.shape[0] else tensor.landmarks
    confidence = tensor.frame_confidence[valid] if valid.shape[0] == tensor.frame_confidence.shape[0] else tensor.frame_confidence
    if landmarks.shape[0] < 2:
        return {
            "dx": 0.0,
            "motion": 0.0,
            "scale_delta": 0.0,
            "index_middle_min": 1.0,
            "confidence": float(np.mean(confidence)) if confidence.size else 0.0,
        }

    wrist = landmarks[:, 0, :2]
    centroid = landmarks[:, :, :2].mean(axis=1)
    scale = palm_scale(landmarks)
    index_middle_distance = np.linalg.norm(landmarks[:, 8, :2] - landmarks[:, 12, :2], axis=1)
    return {
        "dx": float(wrist[-1, 0] - wrist[0, 0]),
        "motion": float(np.linalg.norm(centroid[-1] - centroid[0])),
        "scale_delta": float(scale[-1] - scale[0]),
        "index_middle_min": float(np.min(index_middle_distance)),
        "confidence": float(np.mean(confidence)) if confidence.size else 1.0,
    }
