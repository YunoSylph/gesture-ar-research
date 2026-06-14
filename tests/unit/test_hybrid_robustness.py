from __future__ import annotations

import numpy as np

from research_pipeline.data.tensors import LandmarkTensor
from research_pipeline.evaluation.robustness import PerturbationConfig, perturb_tensor
from research_pipeline.models.common import prediction_from_scores
from research_pipeline.models.hybrid import HybridConfig, HybridRecognizer
from research_pipeline.models.rule_based import RuleBasedRecognizer


def _tensor() -> LandmarkTensor:
    landmarks = np.zeros((12, 21, 3), dtype=np.float32)
    landmarks[:, :, 0] = np.linspace(0.35, 0.55, 12, dtype=np.float32)[:, None]
    landmarks[:, :, 1] = 0.45
    landmarks[:, 5, 0] += 0.04
    landmarks[:, 17, 0] -= 0.04
    landmarks[:, 9, 1] += 0.08
    return LandmarkTensor(
        landmarks=landmarks,
        sequence_mask=np.ones((12,), dtype=bool),
        frame_confidence=np.ones((12,), dtype=np.float32),
        handedness_score=np.ones((12,), dtype=np.float32),
    )


def test_perturb_tensor_preserves_shape() -> None:
    tensor = _tensor()
    rng = np.random.default_rng(7)
    perturbed = perturb_tensor(tensor, PerturbationConfig(name="noise", kind="gaussian_noise", sigma=0.01), rng)
    assert perturbed.landmarks.shape == tensor.landmarks.shape
    assert perturbed.sequence_mask.shape == tensor.sequence_mask.shape
    assert perturbed.frame_confidence.shape == tensor.frame_confidence.shape


def test_hybrid_safety_gate_can_restore_no_gesture() -> None:
    class FakePredictor:
        def predict(self, tensor: LandmarkTensor):
            return prediction_from_scores({"click_2f": 0.42, "no_gesture": 0.40})

    recognizer = HybridRecognizer.__new__(HybridRecognizer)
    recognizer.neural = FakePredictor()
    recognizer.rule = RuleBasedRecognizer()
    recognizer.config = HybridConfig(neural_weight=1.0, geometry_weight=0.0, action_threshold=0.6)
    prediction = recognizer.predict(_tensor())
    assert prediction.label == "no_gesture"


def test_hybrid_can_disable_safety_gate() -> None:
    class FakePredictor:
        def predict(self, tensor: LandmarkTensor):
            return prediction_from_scores({"click_2f": 0.42, "no_gesture": 0.40})

    recognizer = HybridRecognizer.__new__(HybridRecognizer)
    recognizer.neural = FakePredictor()
    recognizer.rule = RuleBasedRecognizer()
    recognizer.config = HybridConfig(neural_weight=1.0, geometry_weight=0.0, enable_safety_gate=False, action_threshold=0.6)
    prediction = recognizer.predict(_tensor())
    assert prediction.label == "click_2f"
