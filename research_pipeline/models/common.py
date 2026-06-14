from __future__ import annotations

from dataclasses import dataclass

from research_pipeline.labels import TARGET_LABELS


@dataclass(slots=True)
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(0.0, value) for value in scores.values()))
    if total <= 0.0:
        return {label: 1.0 / len(TARGET_LABELS) for label in TARGET_LABELS}
    return {label: max(0.0, scores.get(label, 0.0)) / total for label in TARGET_LABELS}


def prediction_from_scores(scores: dict[str, float]) -> Prediction:
    normalized = normalize_scores(scores)
    label = max(normalized, key=normalized.get)
    return Prediction(label=label, confidence=float(normalized[label]), scores=normalized)

