from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research_pipeline.data.tensors import LandmarkTensor
from research_pipeline.labels import TARGET_LABELS
from research_pipeline.models.artifacts import load_artifact
from research_pipeline.models.calibrated import CalibratedFusionConfig, calibrated_fusion_prediction
from research_pipeline.models.common import Prediction, prediction_from_scores
from research_pipeline.models.hybrid import (
    CachedArtifactPredictor,
    GeometryPriorRecognizer,
    HybridConfig,
    fuse_hybrid_predictions,
)


@dataclass(slots=True)
class C6EnsembleConfig:
    model_paths: list[str] = field(default_factory=list)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    calibration: CalibratedFusionConfig = field(default_factory=CalibratedFusionConfig)


class C6EnsembleRecognizer:
    """Official robust recognizer: augmented neural ensemble plus calibrated geometry fusion."""

    def __init__(self, config: C6EnsembleConfig):
        if not config.model_paths:
            raise ValueError("C6EnsembleRecognizer requires at least one model path.")
        self.config = config
        self.predictors = [CachedArtifactPredictor(load_artifact(path)) for path in config.model_paths]
        self.geometry = GeometryPriorRecognizer(config.hybrid)

    def predict(self, tensor: LandmarkTensor) -> Prediction:
        neural = self._ensemble_neural_prediction(tensor)
        geometry = self.geometry.predict(tensor)
        c3 = fuse_hybrid_predictions(neural, geometry, tensor, self.config.hybrid)
        return calibrated_fusion_prediction(neural.scores, c3.scores, self.config.calibration)

    def _ensemble_neural_prediction(self, tensor: LandmarkTensor) -> Prediction:
        scores = {label: 0.0 for label in TARGET_LABELS}
        for predictor in self.predictors:
            prediction = predictor.predict(tensor)
            for label in TARGET_LABELS:
                scores[label] += float(prediction.scores.get(label, 0.0))
        scale = 1.0 / len(self.predictors)
        return prediction_from_scores({label: value * scale for label, value in scores.items()})


def c6_config_from_mapping(payload: dict[str, Any]) -> C6EnsembleConfig:
    calibration_payload = payload.get("calibration", {})
    return C6EnsembleConfig(
        model_paths=[str(path) for path in payload.get("model_paths", [])],
        hybrid=HybridConfig(**payload.get("hybrid", {})),
        calibration=CalibratedFusionConfig(
            c3_weight=float(calibration_payload.get("c3_weight", 0.15)),
            temperature=float(calibration_payload.get("temperature", 1.25)),
            label_biases={
                str(label): float(value)
                for label, value in calibration_payload.get("label_biases", {}).items()
            },
            min_action_confidence=float(calibration_payload.get("min_action_confidence", 0.0)),
            min_action_margin=float(calibration_payload.get("min_action_margin", 0.0)),
        ),
    )
