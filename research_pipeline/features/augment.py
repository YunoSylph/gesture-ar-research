from __future__ import annotations

import numpy as np

from research_pipeline.data.tensors import LandmarkTensor
from research_pipeline.labels import swap_mirrored_label


def mirror_landmarks(tensor: LandmarkTensor, *, image_normalized: bool = True) -> LandmarkTensor:
    landmarks = tensor.landmarks.copy()
    world = tensor.world_landmarks.copy() if tensor.world_landmarks is not None else None
    if image_normalized:
        landmarks[:, :, 0] = 1.0 - landmarks[:, :, 0]
    else:
        landmarks[:, :, 0] = -landmarks[:, :, 0]
    if world is not None:
        world[:, :, 0] = -world[:, :, 0]
    handedness = 1.0 - tensor.handedness_score
    return LandmarkTensor(
        landmarks=landmarks.astype(np.float32),
        sequence_mask=tensor.sequence_mask.copy(),
        frame_confidence=tensor.frame_confidence.copy(),
        handedness_score=handedness.astype(np.float32),
        coord_space=tensor.coord_space,
        world_landmarks=world.astype(np.float32) if world is not None else None,
    )


def mirrored_target_label(label: str) -> str:
    return swap_mirrored_label(label)

