"""Confidence calibration metrics for gesture recognition.

The recognition stack is described as "calibrated", yet the earlier evaluation
only reported accuracy and F1. This module adds the calibration view that a
defensible study needs: expected/maximum calibration error, multi-class Brier
score, and the reliability curve used to plot confidence against accuracy.

The functions operate on a probability matrix (one row per sample, one column
per target label in ``TARGET_LABELS`` order) so they can be reused by any model
path that exposes a score distribution, not only the calibrated fusion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from research_pipeline.labels import TARGET_LABELS, label_to_index


@dataclass(slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float
    weight: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class CalibrationReport:
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    mean_confidence: float
    accuracy: float
    overconfidence: float
    num_bins: int
    num_samples: int
    bins: list[ReliabilityBin]

    def to_dict(self) -> dict:
        return {
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "brier_score": self.brier_score,
            "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy,
            "overconfidence": self.overconfidence,
            "num_bins": self.num_bins,
            "num_samples": self.num_samples,
            "bins": [bin_.to_dict() for bin_ in self.bins],
        }


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D probability matrix, got shape {matrix.shape}.")
    if matrix.shape[1] != len(TARGET_LABELS):
        raise ValueError(
            f"Expected {len(TARGET_LABELS)} class columns, got {matrix.shape[1]}."
        )
    matrix = np.clip(matrix, 0.0, None)
    totals = matrix.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= 0.0
    if np.any(empty):
        matrix = matrix.copy()
        matrix[empty] = 1.0 / len(TARGET_LABELS)
        totals = matrix.sum(axis=1, keepdims=True)
    return matrix / np.clip(totals, 1e-12, None)


def _true_indices(y_true: list[str]) -> np.ndarray:
    return np.array([label_to_index(label) for label in y_true], dtype=np.int64)


def reliability_bins(
    confidences: np.ndarray,
    correct: np.ndarray,
    num_bins: int = 15,
) -> list[ReliabilityBin]:
    """Equal-width binning of top-label confidence against empirical accuracy."""

    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    total = confidences.shape[0]
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins: list[ReliabilityBin] = []
    for index in range(num_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == num_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidences[mask].mean())
            accuracy = float(correct[mask].mean())
        else:
            mean_confidence = 0.0
            accuracy = 0.0
        bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_confidence=mean_confidence,
                accuracy=accuracy,
                weight=float(count / total) if total else 0.0,
            )
        )
    return bins


def brier_score(probabilities: np.ndarray, y_true: list[str]) -> float:
    """Multi-class Brier score: mean squared error against the one-hot target."""

    matrix = _normalize_probabilities(probabilities)
    indices = _true_indices(y_true)
    if matrix.shape[0] != indices.shape[0]:
        raise ValueError("Probability rows and labels must match in length.")
    one_hot = np.zeros_like(matrix)
    one_hot[np.arange(matrix.shape[0]), indices] = 1.0
    return float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1)))


def compute_calibration_report(
    probabilities: np.ndarray,
    y_true: list[str],
    *,
    num_bins: int = 15,
) -> CalibrationReport:
    """Compute ECE/MCE/Brier and the reliability curve for one model output."""

    matrix = _normalize_probabilities(probabilities)
    indices = _true_indices(y_true)
    if matrix.shape[0] != indices.shape[0]:
        raise ValueError("Probability rows and labels must match in length.")
    if matrix.shape[0] == 0:
        return CalibrationReport(
            expected_calibration_error=0.0,
            maximum_calibration_error=0.0,
            brier_score=0.0,
            mean_confidence=0.0,
            accuracy=0.0,
            overconfidence=0.0,
            num_bins=num_bins,
            num_samples=0,
            bins=[],
        )

    predictions = matrix.argmax(axis=1)
    confidences = matrix[np.arange(matrix.shape[0]), predictions]
    correct = (predictions == indices).astype(np.float64)

    bins = reliability_bins(confidences, correct, num_bins=num_bins)
    ece = float(sum(bin_.weight * abs(bin_.accuracy - bin_.mean_confidence) for bin_ in bins))
    populated = [bin_ for bin_ in bins if bin_.count > 0]
    mce = float(max((abs(bin_.accuracy - bin_.mean_confidence) for bin_ in populated), default=0.0))

    mean_confidence = float(confidences.mean())
    accuracy = float(correct.mean())
    return CalibrationReport(
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        brier_score=brier_score(matrix, y_true),
        mean_confidence=mean_confidence,
        accuracy=accuracy,
        overconfidence=float(mean_confidence - accuracy),
        num_bins=num_bins,
        num_samples=int(matrix.shape[0]),
        bins=bins,
    )
