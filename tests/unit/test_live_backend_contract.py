import numpy as np

from research_pipeline.data.tensors import LandmarkTensor
from research_pipeline.models.common import prediction_from_scores
from research_pipeline.serve.live_backend import LiveLandmarkGestureController, LivePredictionStabilizer, camera_status, health, methods


def test_live_backend_exposes_primary_and_ablation_methods() -> None:
    payload = health()
    method_payload = methods()

    assert "c1t_tcn" in payload["methods"]
    assert "c6_ensemble" in payload["methods"]
    assert any(item["id"] == "c6_ensemble" for item in method_payload["methods"])
    assert "c3" in payload["ablation_methods"]
    assert any(item["id"] == "c3" for item in method_payload["ablations"])


def test_live_backend_exposes_task_aware_interaction_mode() -> None:
    assert "c4_task_aware" in health()["interaction_modes"]


def test_camera_status_contract_is_available_without_active_camera() -> None:
    status = camera_status()
    assert {
        "running",
        "error",
        "camera_index",
        "requested_width",
        "requested_height",
        "target_fps",
        "width",
        "height",
        "capture_fps",
        "frame_age_ms",
        "backend",
    }.issubset(status)


def _live_tensor(*, fingertip_distance: float, dx: float = 0.0, scale_delta: float = 0.0) -> LandmarkTensor:
    frame_count = 8
    landmarks = np.zeros((frame_count, 21, 3), dtype=np.float32)
    for index in range(frame_count):
        t = index / max(1, frame_count - 1)
        center_x = 0.5 + dx * t
        scale = 0.2 * (1.0 + scale_delta * t)
        wrist_y = 0.8
        landmarks[index, 0, :2] = np.array([center_x, wrist_y], dtype=np.float32)
        landmarks[index, 5, :2] = np.array([center_x - 0.09, wrist_y - scale * 0.72], dtype=np.float32)
        landmarks[index, 9, :2] = np.array([center_x, wrist_y - scale], dtype=np.float32)
        landmarks[index, 17, :2] = np.array([center_x + 0.09, wrist_y - scale * 0.68], dtype=np.float32)
        landmarks[index, 8, :2] = np.array([center_x - fingertip_distance / 2.0, wrist_y - scale * 1.9], dtype=np.float32)
        landmarks[index, 12, :2] = np.array([center_x + fingertip_distance / 2.0, wrist_y - scale * 1.9], dtype=np.float32)
    return LandmarkTensor(
        landmarks=landmarks,
        sequence_mask=np.ones((landmarks.shape[0],), dtype=bool),
        frame_confidence=np.ones((landmarks.shape[0],), dtype=np.float32),
        handedness_score=np.ones((landmarks.shape[0],), dtype=np.float32),
    )


def test_live_stabilizer_demotes_ambiguous_click() -> None:
    stabilizer = LivePredictionStabilizer()
    tensor = _live_tensor(fingertip_distance=0.04)
    ambiguous_click = prediction_from_scores({"click_2f": 0.54, "point_2f": 0.46})

    labels = [stabilizer.update(ambiguous_click, tensor).label for _ in range(5)]

    assert "click_2f" not in labels


def test_live_stabilizer_rearms_click_after_single_emission() -> None:
    stabilizer = LivePredictionStabilizer()
    tensor = _live_tensor(fingertip_distance=0.04)
    strong_click = prediction_from_scores({"click_2f": 0.90, "point_2f": 0.06, "no_gesture": 0.04})

    labels = [stabilizer.update(strong_click, tensor).label for _ in range(4)]

    assert labels[-2] == "click_2f"
    assert labels[-1] != "click_2f"


def test_live_landmark_controller_promotes_visible_hand_to_point() -> None:
    controller = LiveLandmarkGestureController()
    tensor = _live_tensor(fingertip_distance=0.13)
    raw = prediction_from_scores({"no_gesture": 0.95, "click_2f": 0.05})

    prediction = controller.update(raw, tensor)

    assert prediction.label == "point_2f"


def test_live_landmark_controller_requires_open_before_click() -> None:
    controller = LiveLandmarkGestureController()
    raw = prediction_from_scores({"click_2f": 0.95, "point_2f": 0.05})
    closed = _live_tensor(fingertip_distance=0.04)
    open_hand = _live_tensor(fingertip_distance=0.13)

    assert controller.update(raw, closed).label != "click_2f"
    assert controller.update(raw, open_hand).label == "point_2f"
    assert controller.update(raw, closed).label == "point_2f"
    assert controller.context()["mode"] == "preparing"
    assert controller.update(raw, closed).label == "click_2f"
    assert controller.context()["mode"] == "locked"
    assert controller.update(raw, closed).label == "click_2f"


def test_live_landmark_controller_detects_wide_horizontal_swipe() -> None:
    controller = LiveLandmarkGestureController()
    raw = prediction_from_scores({"no_gesture": 0.80, "swipe_right": 0.20})
    tensor = _live_tensor(fingertip_distance=0.13, dx=0.22)

    assert controller.update(raw, tensor).label == "point_2f"
    prediction = controller.update(raw, tensor)

    assert prediction.label == "swipe_right"


def test_live_landmark_controller_focuses_expected_label() -> None:
    controller = LiveLandmarkGestureController()
    raw = prediction_from_scores({"no_gesture": 0.80, "swipe_right": 0.20})
    tensor = _live_tensor(fingertip_distance=0.13, dx=0.24)

    prediction = controller.update(raw, tensor, expected_label="click_2f")

    assert prediction.label == "point_2f"
    assert controller.context()["expected_label"] == "click_2f"
