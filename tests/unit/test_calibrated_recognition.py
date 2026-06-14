from __future__ import annotations

import numpy as np

from research_pipeline.labels import TARGET_LABELS, label_to_index
from research_pipeline.models.calibrated import CalibratedFusionConfig, calibrated_fusion_labels, calibrated_fusion_matrix


def test_label_bias_can_recover_weak_class_prediction() -> None:
    c1 = np.full((1, len(TARGET_LABELS)), 0.01, dtype=np.float64)
    c3 = c1.copy()
    c1[0, label_to_index("swipe_right")] = 0.48
    c1[0, label_to_index("swipe_left")] = 0.42
    c3[0, label_to_index("swipe_right")] = 0.46
    c3[0, label_to_index("swipe_left")] = 0.43

    config = CalibratedFusionConfig(label_biases={"swipe_left": 0.24})

    assert calibrated_fusion_labels(c1, c3, config) == ["swipe_left"]


def test_action_abstention_prefers_no_gesture_on_low_margin() -> None:
    scores = np.full((1, len(TARGET_LABELS)), 0.01, dtype=np.float64)
    scores[0, label_to_index("click_2f")] = 0.43
    scores[0, label_to_index("no_gesture")] = 0.42

    config = CalibratedFusionConfig(min_action_margin=0.05)
    calibrated = calibrated_fusion_matrix(scores, scores, config)

    assert TARGET_LABELS[int(calibrated.argmax(axis=1)[0])] == "no_gesture"
