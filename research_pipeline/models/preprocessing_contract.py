from __future__ import annotations

from typing import Any

from research_pipeline.data.tensors import LandmarkTensor
from research_pipeline.features.multiview import NUM_LANDMARKS, joint_pair_indices
from research_pipeline.features.preprocessing import (
    INDEX_MCP,
    MIDDLE_MCP,
    PINKY_MCP,
    WRIST,
    preprocess_dual_view,
    resample_landmarks,
)

# Order of the 11 global-motion features as concatenated in preprocess_dual_view.
MOTION_ORDER: tuple[str, ...] = (
    "centroid_x",
    "centroid_y",
    "wrist_x",
    "wrist_y",
    "centroid_dx",
    "centroid_dy",
    "wrist_dx",
    "wrist_dy",
    "hand_size",
    "hand_size_delta",
    "frame_confidence",
)


def feature_layout_contract(
    *, target_length: int = 32, multiview_coords: int = 2, include_multiview: bool = True
) -> dict[str, Any]:
    """Exact, machine-readable layout of the per-frame feature vector.

    Derived from the live feature constants so it cannot drift from
    ``preprocess_dual_view``. This is the spec the iOS/Swift on-device
    preprocessing must reproduce to feed the Core ML model. With
    ``include_multiview`` the JCD + slow/fast motion block (mv models, 326 dims)
    is appended; without it only the dual-view pose + motion block (74 dims).
    """

    pairs = joint_pair_indices(NUM_LANDMARKS)
    blocks: list[dict[str, Any]] = []
    cursor = 0

    def add(name: str, dim: int, **extra: Any) -> None:
        nonlocal cursor
        blocks.append({"name": name, "start": cursor, "end": cursor + dim, "dim": dim, **extra})
        cursor += dim

    add(
        "pose",
        NUM_LANDMARKS * 3,
        coords=3,
        desc="(L[j] - L[WRIST]) / palm_scale; 21 joints row-major as (x,y,z)",
    )
    add(
        "motion",
        len(MOTION_ORDER),
        order=list(MOTION_ORDER),
        desc="global image-plane motion in raw normalized xy (NOT palm-scaled)",
    )
    if include_multiview:
        add(
            "jcd",
            int(pairs.shape[0]),
            coords=multiview_coords,
            pair_indices=pairs.tolist(),
            desc="||L[i].xy - L[j].xy|| / palm_scale for upper-triangle pairs i<j",
        )
        add(
            "slow_motion",
            NUM_LANDMARKS,
            coords=multiview_coords,
            step=1,
            desc="per-joint ||L_t - L_{t-1}|| / palm_scale; first frame is zeros",
        )
        add(
            "fast_motion",
            NUM_LANDMARKS,
            coords=multiview_coords,
            step=2,
            desc="per-joint ||L_t - L_{t-2}|| / palm_scale; first two frames are zeros",
        )

    return {
        "target_length": target_length,
        "multiview_coords": multiview_coords,
        "include_multiview": include_multiview,
        "feature_dim": cursor,
        "landmark_count": NUM_LANDMARKS,
        "coord_space": "image_normalized_xyz",
        "constants": {
            "WRIST": WRIST,
            "INDEX_MCP": INDEX_MCP,
            "PINKY_MCP": PINKY_MCP,
            "MIDDLE_MCP": MIDDLE_MCP,
            "palm_scale_eps": 1e-6,
        },
        "palm_scale": (
            "max(||L[INDEX_MCP].xy - L[PINKY_MCP].xy||, ||L[MIDDLE_MCP].xy - L[WRIST].xy||), "
            "floored at palm_scale_eps"
        ),
        "masking": "after all blocks are computed, frames with sequence_mask==False are set to 0",
        "blocks": blocks,
    }


def golden_sample(
    tensor: LandmarkTensor,
    *,
    sample_id: str,
    target_label: str,
    target_length: int = 32,
    multiview_coords: int = 2,
) -> dict[str, Any]:
    """A landmarks->features golden record for Swift preprocessing parity tests.

    The input is the resampled ``target_length`` window (so the feature math is
    isolated from resampling); the expected output is the full feature matrix.
    """

    sampled = resample_landmarks(tensor, target_length=target_length)
    sequence = preprocess_dual_view(
        tensor,
        target_length=target_length,
        include_multiview=True,
        multiview_coords=multiview_coords,
    )
    return {
        "sample_id": sample_id,
        "target_label": target_label,
        "input": {
            "landmarks": sampled.landmarks.tolist(),
            "sequence_mask": sampled.sequence_mask.astype(bool).tolist(),
            "frame_confidence": sampled.frame_confidence.tolist(),
        },
        "expected_features": sequence.features.tolist(),
    }
