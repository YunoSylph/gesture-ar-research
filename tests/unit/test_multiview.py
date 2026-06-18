from __future__ import annotations

import numpy as np

from research_pipeline.data.tensors import LandmarkTensor
from research_pipeline.features.multiview import (
    NUM_LANDMARKS,
    extract_multiview,
    frame_velocity,
    joint_collection_distances,
    joint_pair_indices,
)


def _tensor(landmarks: np.ndarray, mask: np.ndarray | None = None) -> LandmarkTensor:
    window = landmarks.shape[0]
    if mask is None:
        mask = np.ones((window,), dtype=bool)
    return LandmarkTensor(
        landmarks=landmarks.astype(np.float32),
        sequence_mask=mask.astype(bool),
        frame_confidence=np.ones((window,), dtype=np.float32),
        handedness_score=np.ones((window,), dtype=np.float32),
    )


def _static_hand(window: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5, 0.1, size=(NUM_LANDMARKS, 3)).astype(np.float32)
    return np.repeat(base[None, :, :], window, axis=0)


def test_pair_indices_count_matches_combinations() -> None:
    pairs = joint_pair_indices(NUM_LANDMARKS)
    assert pairs.shape == (NUM_LANDMARKS * (NUM_LANDMARKS - 1) // 2, 2)
    assert (pairs[:, 0] < pairs[:, 1]).all()


def test_feature_shapes_align_to_window() -> None:
    landmarks = _static_hand(window=10)
    features = extract_multiview(_tensor(landmarks))
    pairs = NUM_LANDMARKS * (NUM_LANDMARKS - 1) // 2
    assert features.jcd.shape == (10, pairs)
    assert features.slow_motion.shape == (10, NUM_LANDMARKS)
    assert features.fast_motion.shape == (10, NUM_LANDMARKS)
    assert features.matrix.shape == (10, pairs + 2 * NUM_LANDMARKS)


def test_jcd_translation_invariant() -> None:
    landmarks = _static_hand()
    shifted = landmarks + np.array([0.3, -0.2, 0.1], dtype=np.float32)
    base = joint_collection_distances(landmarks, normalize=False)
    moved = joint_collection_distances(shifted, normalize=False)
    assert np.allclose(base, moved, atol=1e-5)


def test_jcd_rotation_invariant() -> None:
    landmarks = _static_hand()
    theta = 0.7
    rot = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rotated = landmarks @ rot.T
    base = joint_collection_distances(landmarks, normalize=False)
    moved = joint_collection_distances(rotated, normalize=False)
    assert np.allclose(base, moved, atol=1e-4)


def test_jcd_scale_invariant_when_normalized() -> None:
    landmarks = _static_hand()
    scaled = landmarks * 2.5
    base = joint_collection_distances(landmarks, normalize=True)
    moved = joint_collection_distances(scaled, normalize=True)
    assert np.allclose(base, moved, atol=1e-4)


def test_velocity_zero_for_static_sequence() -> None:
    landmarks = _static_hand(window=6)
    slow = frame_velocity(landmarks, step=1)
    assert np.allclose(slow, 0.0, atol=1e-6)


def test_velocity_detects_translation() -> None:
    base = _static_hand(window=6)[0]
    drift = np.arange(6, dtype=np.float32)[:, None, None] * np.array([0.05, 0.0, 0.0], dtype=np.float32)
    landmarks = base[None, :, :] + drift
    slow = frame_velocity(landmarks, step=1, normalize=False)
    # First frame has no defined velocity; later frames share a uniform speed.
    assert np.allclose(slow[0], 0.0)
    assert slow[1:].min() > 0.0
    assert np.allclose(slow[1:], slow[1], atol=1e-5)


def test_fast_motion_uses_two_frame_step() -> None:
    base = _static_hand(window=6)[0]
    drift = np.arange(6, dtype=np.float32)[:, None, None] * np.array([0.0, 0.04, 0.0], dtype=np.float32)
    landmarks = base[None, :, :] + drift
    slow = frame_velocity(landmarks, step=1, normalize=False)
    fast = frame_velocity(landmarks, step=2, normalize=False)
    assert np.allclose(fast[0], 0.0) and np.allclose(fast[1], 0.0)
    # A 2-frame displacement is twice a 1-frame displacement under constant drift.
    assert np.allclose(fast[2:], 2.0 * slow[2:], atol=1e-5)


def test_masked_frames_zeroed() -> None:
    landmarks = _static_hand(window=5)
    mask = np.array([True, True, False, True, True])
    features = extract_multiview(_tensor(landmarks, mask))
    assert np.allclose(features.matrix[2], 0.0)
    assert np.allclose(features.jcd[2], 0.0)


def test_preprocess_appends_multiview_block() -> None:
    from research_pipeline.features.preprocessing import preprocess_dual_view

    landmarks = _static_hand(window=40)
    tensor = _tensor(landmarks)
    base = preprocess_dual_view(tensor, target_length=32)
    fused = preprocess_dual_view(tensor, target_length=32, include_multiview=True, multiview_coords=2)
    pairs = NUM_LANDMARKS * (NUM_LANDMARKS - 1) // 2
    # The dual-view features are unchanged; the multi-view block is appended.
    assert fused.features.shape[1] == base.features.shape[1] + pairs + 2 * NUM_LANDMARKS
    assert np.allclose(fused.features[:, : base.features.shape[1]], base.features)


def test_artifact_feature_flags_default_to_dual_view() -> None:
    from research_pipeline.models.artifacts import artifact_feature_flags

    assert artifact_feature_flags({}) == (False, 2)
    assert artifact_feature_flags(
        {"features": {"include_multiview": True, "multiview_coords": 3}}
    ) == (True, 3)
