from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research_pipeline.data.tensors import LandmarkTensor
from research_pipeline.features.preprocessing import palm_scale, resample_landmarks

NUM_LANDMARKS = 21


@dataclass(slots=True)
class MultiViewFeatures:
    """OO-dMVMT-style multi-view representation of a landmark window.

    The three views follow Cunico et al. 2023 (OO-dMVMT), Sec. 3.1:

    - ``jcd``: flattened Joint Collection Distances (DDNet, Yang et al.) -- the
      pairwise Euclidean distances between every joint pair. Pairwise distances
      are translation- and rotation-invariant; dividing by ``palm_scale`` adds
      scale invariance, so this view encodes hand *pose* independent of where the
      hand is in frame or how close it is to the camera.
    - ``slow_motion``: per-joint speed over a 1-frame step (short-term slow
      motion ``M_slow``).
    - ``fast_motion``: per-joint speed over a 2-frame step (short-term fast
      motion ``M_fast``), which captures quicker dynamics.

    Unlike the paper -- whose encoders consume the raw ``W-1`` / ``W/2-1`` view
    lengths -- every view here is aligned to the window length ``W`` with leading
    zeros for the velocity steps. This keeps a uniform per-frame stream that is
    convenient both for online (per-frame) live control and as input to a
    temporal model. ``matrix`` is the per-frame concatenation of the three views.
    """

    jcd: np.ndarray
    slow_motion: np.ndarray
    fast_motion: np.ndarray
    matrix: np.ndarray
    mask: np.ndarray
    pair_indices: np.ndarray

    @property
    def num_pairs(self) -> int:
        return int(self.pair_indices.shape[0])


def joint_pair_indices(num_joints: int = NUM_LANDMARKS) -> np.ndarray:
    """Upper-triangle joint pairs ``(i, j)`` with ``i < j`` -- ``C(num_joints, 2)`` rows."""

    rows, cols = np.triu_indices(num_joints, k=1)
    return np.stack([rows, cols], axis=1).astype(np.int64)


def joint_collection_distances(
    landmarks: np.ndarray,
    *,
    pair_indices: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Pairwise joint distances per frame -> ``(W, C(J, 2))``.

    ``landmarks`` is ``(W, J, D)``. With ``normalize`` the distances are divided
    by ``palm_scale`` (per frame), making the view scale-invariant.
    """

    if landmarks.ndim != 3:
        raise ValueError(f"landmarks must be [W, J, D], got shape {landmarks.shape}.")
    pairs = joint_pair_indices(landmarks.shape[1]) if pair_indices is None else pair_indices
    left = landmarks[:, pairs[:, 0], :]
    right = landmarks[:, pairs[:, 1], :]
    distances = np.linalg.norm(left - right, axis=2).astype(np.float32)
    if normalize:
        if scale is None:
            scale = palm_scale(landmarks)
        distances = distances / np.maximum(scale[:, None], 1e-6)
    return distances.astype(np.float32)


def frame_velocity(
    landmarks: np.ndarray,
    *,
    step: int = 1,
    scale: np.ndarray | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Per-joint speed over a ``step``-frame interval, aligned to ``W`` -> ``(W, J)``.

    The first ``step`` frames have no defined velocity and are filled with zeros,
    so the output stays aligned with the window for per-frame online use.
    """

    if landmarks.ndim != 3:
        raise ValueError(f"landmarks must be [W, J, D], got shape {landmarks.shape}.")
    window = landmarks.shape[0]
    num_joints = landmarks.shape[1]
    velocity = np.zeros((window, num_joints), dtype=np.float32)
    if window > step:
        displacement = landmarks[step:] - landmarks[:-step]
        speed = np.linalg.norm(displacement, axis=2).astype(np.float32)
        if normalize:
            if scale is None:
                scale = palm_scale(landmarks)
            speed = speed / np.maximum(scale[step:, None], 1e-6)
        velocity[step:] = speed
    return velocity


def extract_multiview(
    tensor: LandmarkTensor,
    *,
    target_length: int | None = None,
    normalize: bool = True,
    coords: int = 3,
) -> MultiViewFeatures:
    """Build the OO-dMVMT multi-view representation from a landmark window.

    ``coords`` selects how many coordinate dimensions feed the geometry/motion
    views (2 = image xy only, 3 = include the relative-depth channel). MediaPipe
    image ``z`` is noisy, so callers that only have image landmarks may prefer
    ``coords=2``.
    """

    source = resample_landmarks(tensor, target_length=target_length) if target_length else tensor
    landmarks = source.landmarks.astype(np.float32)[:, :, :coords]
    mask = source.sequence_mask.astype(bool)
    scale = palm_scale(source.landmarks)
    pairs = joint_pair_indices(landmarks.shape[1])

    jcd = joint_collection_distances(landmarks, pair_indices=pairs, scale=scale, normalize=normalize)
    slow = frame_velocity(landmarks, step=1, scale=scale, normalize=normalize)
    fast = frame_velocity(landmarks, step=2, scale=scale, normalize=normalize)

    matrix = np.concatenate([jcd, slow, fast], axis=1).astype(np.float32)
    if mask.shape[0] == matrix.shape[0]:
        matrix[~mask] = 0.0
        jcd = jcd.copy()
        slow = slow.copy()
        fast = fast.copy()
        jcd[~mask] = 0.0
        slow[~mask] = 0.0
        fast[~mask] = 0.0
    return MultiViewFeatures(
        jcd=jcd,
        slow_motion=slow,
        fast_motion=fast,
        matrix=matrix,
        mask=mask,
        pair_indices=pairs,
    )
