from __future__ import annotations

import numpy as np

from research_pipeline.data.tensors import LandmarkTensor
from research_pipeline.labels import validate_target_label
from research_pipeline.utils.random import stable_rng


BASE_HAND = np.array(
    [
        [0.00, 0.00, 0.00],
        [-0.05, -0.04, 0.00],
        [-0.08, -0.09, 0.00],
        [-0.10, -0.14, 0.00],
        [-0.12, -0.19, 0.00],
        [-0.03, -0.11, 0.00],
        [-0.03, -0.18, 0.00],
        [-0.03, -0.25, 0.00],
        [-0.03, -0.32, 0.00],
        [0.02, -0.12, 0.00],
        [0.02, -0.20, 0.00],
        [0.02, -0.29, 0.00],
        [0.02, -0.37, 0.00],
        [0.07, -0.10, 0.00],
        [0.08, -0.17, 0.00],
        [0.08, -0.24, 0.00],
        [0.08, -0.30, 0.00],
        [0.12, -0.07, 0.00],
        [0.14, -0.13, 0.00],
        [0.15, -0.19, 0.00],
        [0.16, -0.25, 0.00],
    ],
    dtype=np.float32,
)


def synthetic_landmarks(label: str, *, length: int = 32, seed: int = 0, sample_id: str = "") -> LandmarkTensor:
    validate_target_label(label)
    rng = stable_rng(seed, f"{sample_id}:{label}")
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    center = np.column_stack(
        [
            np.full(length, 0.5, dtype=np.float32),
            np.full(length, 0.58, dtype=np.float32),
            np.zeros(length, dtype=np.float32),
        ]
    )
    scale = np.full(length, 0.55, dtype=np.float32)

    if label == "swipe_left":
        center[:, 0] += 0.24 * (0.5 - t)
    elif label == "swipe_right":
        center[:, 0] += 0.24 * (t - 0.5)
    elif label == "zoom_in":
        scale += 0.22 * t
    elif label == "zoom_out":
        scale += 0.22 * (1.0 - t)
    elif label == "click_2f":
        pulse = np.exp(-((t - 0.55) ** 2) / 0.01).astype(np.float32)
        scale -= 0.05 * pulse
    elif label == "point_2f":
        center[:, 1] -= 0.03 * np.sin(t * np.pi).astype(np.float32)
    else:
        center[:, 0] += 0.01 * np.sin(t * np.pi * 2.0).astype(np.float32)

    landmarks = BASE_HAND[None, :, :] * scale[:, None, None] + center[:, None, :]
    if label == "click_2f":
        # Move index/middle fingertips closer for a short confirmation pulse.
        pulse = np.exp(-((t - 0.55) ** 2) / 0.008).astype(np.float32)
        landmarks[:, 8, :2] += pulse[:, None] * np.array([0.04, 0.05], dtype=np.float32)
        landmarks[:, 12, :2] += pulse[:, None] * np.array([-0.04, 0.05], dtype=np.float32)
    noise = rng.normal(0.0, 0.003, size=landmarks.shape).astype(np.float32)
    landmarks = np.clip(landmarks + noise, 0.0, 1.0).astype(np.float32)
    confidence = np.clip(rng.normal(0.95, 0.02, size=(length,)), 0.75, 1.0).astype(np.float32)
    return LandmarkTensor(
        landmarks=landmarks,
        sequence_mask=np.ones((length,), dtype=bool),
        frame_confidence=confidence,
        handedness_score=np.ones((length,), dtype=np.float32),
        coord_space="image_normalized_xyz",
    )

