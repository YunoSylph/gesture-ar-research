from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research_pipeline.labels import TARGET_LABELS


@dataclass(slots=True)
class RecognitionMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }


def compute_recognition_metrics(y_true: list[str], y_pred: list[str]) -> RecognitionMetrics:
    labels = list(TARGET_LABELS)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[index[true], index[pred]] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_values = []
    recalls = []
    weights = []
    for i, label in enumerate(labels):
        tp = float(matrix[i, i])
        fp = float(matrix[:, i].sum() - matrix[i, i])
        fn = float(matrix[i, :].sum() - matrix[i, i])
        support = float(matrix[i, :].sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        recalls.append(recall)
        weights.append(support)
    total = float(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    weights_array = np.array(weights, dtype=np.float64)
    weighted_f1 = float(np.average(f1_values, weights=weights_array)) if weights_array.sum() else 0.0
    active_recalls = [recall for recall, support in zip(recalls, weights) if support > 0]
    return RecognitionMetrics(
        accuracy=accuracy,
        macro_f1=float(np.mean(f1_values)) if f1_values else 0.0,
        weighted_f1=weighted_f1,
        balanced_accuracy=float(np.mean(active_recalls)) if active_recalls else 0.0,
        per_class=per_class,
        confusion_matrix=matrix.astype(int).tolist(),
    )

