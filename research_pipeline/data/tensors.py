from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research_pipeline.utils.errors import SchemaError


@dataclass(slots=True)
class LandmarkTensor:
    landmarks: np.ndarray
    sequence_mask: np.ndarray
    frame_confidence: np.ndarray
    handedness_score: np.ndarray
    coord_space: str = "image_normalized_xyz"
    world_landmarks: np.ndarray | None = None


def validate_landmark_tensor(tensor: LandmarkTensor) -> None:
    if tensor.landmarks.ndim != 3 or tensor.landmarks.shape[1:] != (21, 3):
        raise SchemaError(f"landmarks must have shape [T,21,3], got {tensor.landmarks.shape}.")
    t = tensor.landmarks.shape[0]
    if tensor.sequence_mask.shape != (t,):
        raise SchemaError("sequence_mask must have shape [T].")
    if tensor.frame_confidence.shape != (t,):
        raise SchemaError("frame_confidence must have shape [T].")
    if tensor.handedness_score.shape not in {(t,), (1,)}:
        raise SchemaError("handedness_score must be scalar-like or shape [T].")
    if tensor.world_landmarks is not None and tensor.world_landmarks.shape != tensor.landmarks.shape:
        raise SchemaError("world_landmarks must match landmarks shape.")


def save_landmark_npz(path: str | Path, tensor: LandmarkTensor, **metadata: Any) -> None:
    validate_landmark_tensor(tensor)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "landmarks": tensor.landmarks.astype(np.float32),
        "sequence_mask": tensor.sequence_mask.astype(bool),
        "frame_confidence": tensor.frame_confidence.astype(np.float32),
        "handedness_score": tensor.handedness_score.astype(np.float32),
        "coord_space": np.array(tensor.coord_space),
    }
    if tensor.world_landmarks is not None:
        payload["world_landmarks"] = tensor.world_landmarks.astype(np.float32)
    for key, value in metadata.items():
        payload[f"meta_{key}"] = np.array(value)
    np.savez_compressed(output, **payload)


def load_landmark_npz(path: str | Path) -> LandmarkTensor:
    with np.load(Path(path), allow_pickle=False) as data:
        world = data["world_landmarks"] if "world_landmarks" in data else None
        tensor = LandmarkTensor(
            landmarks=data["landmarks"].astype(np.float32),
            sequence_mask=data["sequence_mask"].astype(bool),
            frame_confidence=data["frame_confidence"].astype(np.float32),
            handedness_score=data["handedness_score"].astype(np.float32),
            coord_space=str(data["coord_space"].item()) if "coord_space" in data else "image_normalized_xyz",
            world_landmarks=world.astype(np.float32) if world is not None else None,
        )
    validate_landmark_tensor(tensor)
    return tensor

