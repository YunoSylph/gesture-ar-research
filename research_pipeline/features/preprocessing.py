from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research_pipeline.data.tensors import LandmarkTensor


WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17
MIDDLE_MCP = 9


@dataclass(slots=True)
class PreprocessedSequence:
    pose: np.ndarray
    motion: np.ndarray
    mask: np.ndarray
    features: np.ndarray


def resample_indices(length: int, target_length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((target_length,), dtype=np.int64)
    if length == target_length:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, target_length).round().astype(np.int64)


def resample_landmarks(tensor: LandmarkTensor, target_length: int = 32) -> LandmarkTensor:
    idx = resample_indices(tensor.landmarks.shape[0], target_length)
    handedness = tensor.handedness_score
    if handedness.shape == (1,):
        handedness = np.repeat(handedness, target_length)
    else:
        handedness = handedness[idx]
    world = tensor.world_landmarks[idx] if tensor.world_landmarks is not None else None
    return LandmarkTensor(
        landmarks=tensor.landmarks[idx].astype(np.float32),
        sequence_mask=tensor.sequence_mask[idx].astype(bool),
        frame_confidence=tensor.frame_confidence[idx].astype(np.float32),
        handedness_score=handedness.astype(np.float32),
        coord_space=tensor.coord_space,
        world_landmarks=world.astype(np.float32) if world is not None else None,
    )


def palm_scale(landmarks: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a = landmarks[:, INDEX_MCP, :2]
    b = landmarks[:, PINKY_MCP, :2]
    c = landmarks[:, WRIST, :2]
    width = np.linalg.norm(a - b, axis=1)
    length = np.linalg.norm(landmarks[:, MIDDLE_MCP, :2] - c, axis=1)
    scale = np.maximum(width, length)
    return np.maximum(scale, eps).astype(np.float32)


def preprocess_dual_view(
    tensor: LandmarkTensor,
    target_length: int = 32,
    *,
    include_multiview: bool = False,
    multiview_coords: int = 2,
) -> PreprocessedSequence:
    """Return pose-normalized and global-motion streams without erasing trajectory.

    With ``include_multiview`` the OO-dMVMT multi-view block (Joint Collection
    Distances plus slow/fast per-joint motion) is appended to the per-frame
    feature vector. The geometry/motion views run on the first ``multiview_coords``
    coordinate channels (2 = image xy, which avoids MediaPipe's noisy relative-z).
    The flag must match between training and inference, so it is recorded in the
    model artifact and threaded back through the predictors.
    """

    sampled = resample_landmarks(tensor, target_length=target_length)
    landmarks = sampled.landmarks.astype(np.float32)
    mask = sampled.sequence_mask.astype(bool)
    wrist = landmarks[:, WRIST : WRIST + 1, :]
    scale = palm_scale(landmarks)[:, None, None]
    pose = (landmarks - wrist) / scale

    centroid = landmarks[:, :, :2].mean(axis=1)
    wrist_xy = landmarks[:, WRIST, :2]
    centroid_delta = np.vstack([np.zeros((1, 2), dtype=np.float32), np.diff(centroid, axis=0)])
    wrist_delta = np.vstack([np.zeros((1, 2), dtype=np.float32), np.diff(wrist_xy, axis=0)])
    velocity = np.concatenate([centroid_delta, wrist_delta], axis=1)
    hand_size = palm_scale(landmarks)[:, None]
    hand_size_delta = np.vstack([np.zeros((1, 1), dtype=np.float32), np.diff(hand_size, axis=0)])
    confidence = sampled.frame_confidence[:, None]
    motion = np.concatenate([centroid, wrist_xy, velocity, hand_size, hand_size_delta, confidence], axis=1)

    pose_flat = pose.reshape(target_length, -1)
    feature_blocks = [pose_flat, motion]
    if include_multiview:
        # Local import avoids a circular dependency: multiview imports palm_scale
        # and resample_landmarks from this module.
        from research_pipeline.features.multiview import extract_multiview

        multiview = extract_multiview(sampled, target_length=None, normalize=True, coords=multiview_coords)
        feature_blocks.append(multiview.matrix)
    features = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    features[~mask] = 0.0
    return PreprocessedSequence(
        pose=pose_flat.astype(np.float32),
        motion=motion.astype(np.float32),
        mask=mask,
        features=features,
    )


def clip_feature_summary(sequence: PreprocessedSequence) -> np.ndarray:
    valid = sequence.features[sequence.mask]
    if valid.size == 0:
        valid = sequence.features
    mean = valid.mean(axis=0)
    std = valid.std(axis=0)
    first = valid[0]
    last = valid[-1]
    delta = last - first
    return np.concatenate([mean, std, delta], axis=0).astype(np.float32)

