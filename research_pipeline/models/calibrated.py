from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from research_pipeline.labels import TARGET_LABELS, label_to_index
from research_pipeline.models.common import Prediction, prediction_from_scores


@dataclass(slots=True)
class CalibratedFusionConfig:
    c3_weight: float = 0.0
    temperature: float = 1.0
    label_biases: dict[str, float] = field(default_factory=dict)
    min_action_confidence: float = 0.0
    min_action_margin: float = 0.0


def calibrated_fusion_prediction(
    c1_scores: Mapping[str, float],
    c3_scores: Mapping[str, float],
    config: CalibratedFusionConfig | None = None,
) -> Prediction:
    config = config or CalibratedFusionConfig()
    c1_matrix = _score_dict_to_matrix(c1_scores)
    c3_matrix = _score_dict_to_matrix(c3_scores)
    calibrated = calibrated_fusion_matrix(c1_matrix, c3_matrix, config)
    scores = {label: float(calibrated[0, index]) for index, label in enumerate(TARGET_LABELS)}
    return prediction_from_scores(scores)


def calibrated_fusion_matrix(
    c1_scores: np.ndarray,
    c3_scores: np.ndarray,
    config: CalibratedFusionConfig | None = None,
) -> np.ndarray:
    config = config or CalibratedFusionConfig()
    c1 = _ensure_2d_probabilities(c1_scores)
    c3 = _ensure_2d_probabilities(c3_scores)
    if c1.shape != c3.shape:
        raise ValueError(f"C1 and C3 score matrices must share shape, got {c1.shape} and {c3.shape}.")
    if c1.shape[1] != len(TARGET_LABELS):
        raise ValueError(f"Expected {len(TARGET_LABELS)} class columns, got {c1.shape[1]}.")

    weight = float(np.clip(config.c3_weight, 0.0, 1.0))
    blended = (1.0 - weight) * c1 + weight * c3
    logits = np.log(np.clip(blended, 1e-9, 1.0)) / max(1e-6, float(config.temperature))
    logits += _bias_vector(config.label_biases)
    probabilities = _softmax(logits)
    return _apply_action_abstention(probabilities, config)


def calibrated_fusion_labels(
    c1_scores: np.ndarray,
    c3_scores: np.ndarray,
    config: CalibratedFusionConfig | None = None,
) -> list[str]:
    probabilities = calibrated_fusion_matrix(c1_scores, c3_scores, config)
    indices = probabilities.argmax(axis=1)
    return [TARGET_LABELS[int(index)] for index in indices]


def _score_dict_to_matrix(scores: Mapping[str, float]) -> np.ndarray:
    return np.array([[float(scores.get(label, 0.0)) for label in TARGET_LABELS]], dtype=np.float64)


def _ensure_2d_probabilities(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D score matrix, got shape {matrix.shape}.")
    matrix = np.clip(matrix, 0.0, None)
    totals = matrix.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= 0.0
    if np.any(empty):
        matrix = matrix.copy()
        matrix[empty] = 1.0 / len(TARGET_LABELS)
        totals = matrix.sum(axis=1, keepdims=True)
    return matrix / np.clip(totals, 1e-12, None)


def _bias_vector(label_biases: Mapping[str, float]) -> np.ndarray:
    vector = np.zeros((len(TARGET_LABELS),), dtype=np.float64)
    for label, bias in label_biases.items():
        if label in TARGET_LABELS:
            vector[label_to_index(label)] = float(bias)
    return vector


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _apply_action_abstention(probabilities: np.ndarray, config: CalibratedFusionConfig) -> np.ndarray:
    min_confidence = max(0.0, float(config.min_action_confidence))
    min_margin = max(0.0, float(config.min_action_margin))
    if min_confidence <= 0.0 and min_margin <= 0.0:
        return probabilities

    no_index = label_to_index("no_gesture")
    adjusted = probabilities.copy()
    top_indices = adjusted.argmax(axis=1)
    top_scores = adjusted[np.arange(adjusted.shape[0]), top_indices]
    masked = adjusted.copy()
    masked[np.arange(adjusted.shape[0]), top_indices] = -np.inf
    second_scores = masked.max(axis=1)
    margins = top_scores - second_scores
    abstain = (top_indices != no_index) & ((top_scores < min_confidence) | (margins < min_margin))
    if not np.any(abstain):
        return adjusted

    adjusted[abstain, no_index] = np.maximum(adjusted[abstain, no_index], top_scores[abstain] + 1e-6)
    adjusted /= np.clip(adjusted.sum(axis=1, keepdims=True), 1e-12, None)
    return adjusted
