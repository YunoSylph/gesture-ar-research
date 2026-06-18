from __future__ import annotations

import numpy as np
import pytest

from research_pipeline.evaluation.calibration import (
    brier_score,
    compute_calibration_report,
    reliability_bins,
)
from research_pipeline.labels import TARGET_LABELS, label_to_index

NUM_LABELS = len(TARGET_LABELS)


def _one_hot(labels: list[str]) -> np.ndarray:
    matrix = np.zeros((len(labels), NUM_LABELS), dtype=np.float64)
    for row, label in enumerate(labels):
        matrix[row, label_to_index(label)] = 1.0
    return matrix


def test_perfectly_calibrated_confident_correct_has_zero_error() -> None:
    labels = list(TARGET_LABELS) * 4
    probabilities = _one_hot(labels)
    report = compute_calibration_report(probabilities, labels)
    assert report.accuracy == pytest.approx(1.0)
    assert report.mean_confidence == pytest.approx(1.0)
    assert report.expected_calibration_error == pytest.approx(0.0)
    assert report.maximum_calibration_error == pytest.approx(0.0)
    assert report.brier_score == pytest.approx(0.0)
    assert report.overconfidence == pytest.approx(0.0)


def test_overconfident_model_has_positive_ece() -> None:
    # Always predicts no_gesture with confidence 0.9 but is right only half the time.
    n = 100
    probabilities = np.full((n, NUM_LABELS), 0.1 / (NUM_LABELS - 1), dtype=np.float64)
    probabilities[:, label_to_index("no_gesture")] = 0.9
    labels = ["no_gesture"] * (n // 2) + ["point_2f"] * (n // 2)
    report = compute_calibration_report(probabilities, labels)
    assert report.accuracy == pytest.approx(0.5)
    assert report.mean_confidence == pytest.approx(0.9)
    # confidence 0.9 vs accuracy 0.5 -> gap of 0.4 in the single populated bin.
    assert report.expected_calibration_error == pytest.approx(0.4, abs=1e-6)
    assert report.maximum_calibration_error == pytest.approx(0.4, abs=1e-6)
    assert report.overconfidence == pytest.approx(0.4, abs=1e-6)


def test_reliability_bins_partition_all_samples() -> None:
    rng = np.random.default_rng(0)
    confidences = rng.uniform(0.0, 1.0, size=500)
    correct = (rng.uniform(size=500) < confidences).astype(float)
    bins = reliability_bins(confidences, correct, num_bins=15)
    assert sum(bin_.count for bin_ in bins) == 500
    assert sum(bin_.weight for bin_ in bins) == pytest.approx(1.0)


def test_brier_score_matches_manual_value() -> None:
    labels = ["no_gesture", "point_2f"]
    probabilities = _one_hot(labels).copy()
    # Soften the first row so its squared error is non-zero and known.
    probabilities[0] = 0.0
    probabilities[0, label_to_index("no_gesture")] = 0.7
    probabilities[0, label_to_index("point_2f")] = 0.3
    # Row 0 error: (1-0.7)^2 + (0-0.3)^2 = 0.18; row 1 (perfect) = 0.
    assert brier_score(probabilities, labels) == pytest.approx(0.18 / 2)


def test_wrong_column_count_raises() -> None:
    with pytest.raises(ValueError):
        compute_calibration_report(np.ones((3, NUM_LABELS + 1)), ["no_gesture"] * 3)


def test_empty_input_is_safe() -> None:
    report = compute_calibration_report(np.zeros((0, NUM_LABELS)), [])
    assert report.num_samples == 0
    assert report.expected_calibration_error == 0.0
