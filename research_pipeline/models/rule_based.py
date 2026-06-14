from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research_pipeline.data.tensors import LandmarkTensor
from research_pipeline.features.preprocessing import preprocess_dual_view, palm_scale
from research_pipeline.models.common import Prediction, prediction_from_scores


@dataclass(slots=True)
class RuleBasedRecognizer:
    swipe_threshold: float = 0.11
    zoom_threshold: float = 0.08
    click_distance_threshold: float = 0.035
    no_motion_threshold: float = 0.025

    def predict(self, tensor: LandmarkTensor) -> Prediction:
        sequence = preprocess_dual_view(tensor, target_length=min(32, max(2, tensor.landmarks.shape[0])))
        valid = sequence.mask
        landmarks = tensor.landmarks[valid] if valid.shape[0] == tensor.landmarks.shape[0] else tensor.landmarks
        if landmarks.shape[0] < 2:
            return prediction_from_scores({"no_gesture": 1.0})

        wrist = landmarks[:, 0, :2]
        centroid = landmarks[:, :, :2].mean(axis=1)
        dx = float(wrist[-1, 0] - wrist[0, 0])
        dy = float(wrist[-1, 1] - wrist[0, 1])
        motion = float(np.linalg.norm(centroid[-1] - centroid[0]))
        scale = palm_scale(landmarks)
        scale_delta = float(scale[-1] - scale[0])
        index_middle_distance = float(np.min(np.linalg.norm(landmarks[:, 8, :2] - landmarks[:, 12, :2], axis=1)))

        scores = {
            "no_gesture": max(0.0, self.no_motion_threshold - motion) + 0.05,
            "swipe_right": max(0.0, dx - self.swipe_threshold) * 8.0,
            "swipe_left": max(0.0, -dx - self.swipe_threshold) * 8.0,
            "zoom_in": max(0.0, scale_delta - self.zoom_threshold) * 10.0,
            "zoom_out": max(0.0, -scale_delta - self.zoom_threshold) * 10.0,
            "click_2f": max(0.0, self.click_distance_threshold - index_middle_distance) * 20.0,
            "point_2f": max(0.0, 0.08 - abs(dy)) * 0.5,
        }
        return prediction_from_scores(scores)

